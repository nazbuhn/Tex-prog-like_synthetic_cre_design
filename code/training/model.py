from pathlib import Path
import shutil
import tempfile
import zipfile
from util import need, sha


def build(outputs, task, archive=None, expected=None):
    import keras
    import crested.tl.zoo
    base = crested.tl.zoo.borzoi(seq_len=2048, target_length=64, num_classes=2608)
    if archive is not None:
        need(sha(archive) == expected, 'Official pretrained archive changed')
        with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.weights.h5'
            with z.open('model.weights.h5') as a, path.open('wb') as b:
                shutil.copyfileobj(a, b)
            base.load_weights(path)
    x = keras.layers.Flatten(name='panel_flatten')(base.get_layer('final_conv_activation').output)
    out = keras.layers.Dense(outputs, activation='softplus' if task == 'count' else None, name='panel_output')(x)
    model = keras.Model(base.inputs, out, name='Borzoi_target_panel')
    need(model.output_shape == (None, outputs) and model.input_shape[1:] == (2048, 4), 'Model shape differs')
    return model


def stable_count_loss(target, prediction, multiplier=1.0, max_weight=100.0):
    """CREsted CosineMSELogLoss, with an rsqrt-safe cosine norm.

    Applies the native dynamic cosine weight clipped to [1, max_weight].
    Nonnegative model predictions and measured accessibility targets only.
    """
    import torch
    a, b = target.float(), prediction.float()
    eps2 = 1e-14
    an = a * torch.rsqrt(torch.clamp(torch.sum(a.square(), dim=-1, keepdim=True), min=eps2))
    bn = b * torch.rsqrt(torch.clamp(torch.sum(b.square(), dim=-1, keepdim=True), min=eps2))
    mse = (torch.log1p(multiplier * a) - torch.log1p(multiplier * b)).square().mean()
    weight = torch.clamp(torch.abs(mse), min=1.0, max=max_weight)
    cosine = -(an * bn).sum(dim=-1)
    return (weight * cosine + mse).mean()


def native_numeric_check(folder):
    import importlib.metadata
    import torch
    import crested
    from util import dump
    native = crested.tl.default_configs('peak_regression_count').loss
    multiplier, max_weight = float(native.multiplier), float(native.max_weight)
    need(multiplier == 1.0 and max_weight == 100.0,
         f'Unexpected CREsted count configuration: multiplier={multiplier}, max_weight={max_weight}')
    # Compare against the exact installed instance, not an assumed default.
    # Include examples with MSE below 1, between 1 and 100, and above 100.
    examples = [
        ([[1., 2., 3.], [0., 1., 0.]], [[2., 3., 4.], [.1, .8, .2]]),
        ([[1., 2., 3.], [0., 1., 0.]], [[10., 20., 40.], [2., 10., 3.]]),
        ([[1., 2., 3.], [0., 1., 0.]], [[1e7, 2e7, 4e7], [3e7, 1e7, 2e7]]),
    ]
    for y, p in examples:
        target = torch.tensor(y, dtype=torch.float32)
        ordinary = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        a = stable_count_loss(target, ordinary, multiplier=multiplier, max_weight=max_weight)
        b = native(target, ordinary)
        need(torch.allclose(a, b, rtol=2e-5, atol=2e-5),
             f'Stable and native losses differ: stable={a.item()}, native={b.item()}')
        grad_a = torch.autograd.grad(a, ordinary, retain_graph=True)[0]
        grad_b = torch.autograd.grad(b, ordinary)[0]
        need(torch.allclose(grad_a, grad_b, rtol=2e-4, atol=2e-5),
             f'Ordinary-input loss gradients differ: max_abs={(grad_a-grad_b).abs().max().item()}')
    for values in [[[0., 0., 0.], [1e-28, 1e-28, 1e-28]], [[1e-12, 0., 0.], [0., 1e-30, 0.]]]:
        p = torch.tensor(values, requires_grad=True)
        loss = stable_count_loss(torch.tensor([[1., 2., 3.], [0., 1., 0.]]), p,
                                 multiplier=multiplier, max_weight=max_weight)
        loss.backward()
        need(torch.isfinite(loss).item() and torch.isfinite(p.grad).all().item(),
             'Tiny/zero predictions cause invalid gradients')
    dump(Path(folder) / 'NATIVE_NUMERIC_CHECK.json', dict(status='PASS',
         ordinary_forward_and_gradient_equivalence=True,
         zero_and_tiny_prediction_backward_finite=True, norm_floor_before_reciprocal_sqrt=True,
         environment_modified=False, multiplier=multiplier, max_weight=max_weight, norm_floor=1e-7,
         versions={k: importlib.metadata.version(k) for k in ['torch', 'keras', 'crested', 'numpy']}))


def save(model, folder, name):
    folder = Path(folder)
    tmp = folder / (name + '.partial.weights.h5')
    model.save_weights(tmp)
    dest = folder / (name + '.weights.h5')
    tmp.replace(dest)
    return sha(dest)
