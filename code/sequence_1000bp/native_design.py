"""Calls CREsted's installed ISE, not a replacement mutation proposer."""
import inspect
import numpy as np
from design_core import insert, dna, DESIGN_LEN, N_SINGLE_MUTANTS
from core import encode, objective, best_step
from io_helpers import need


class ContextAdapter:
    """1000-bp native ISE inputs -> immutable 2048bp contexts -> RC mean FRPM.

    This is a prediction adapter, not a newly fitted Keras model. CREsted
    generates and evaluates all 3,000 substitutions; outputs flatten context x class.
    """
    input_shape = (None, DESIGN_LEN, 4)

    def __init__(self, scorer, contexts, progress_fn=None):
        self.scorer = scorer; self.contexts = contexts
        self.calls = []; self.last_tokens = None; self.progress_fn = progress_fn
        self.last_predictions = None

    def predict(self, onehot, **kwargs):
        x = np.asarray(onehot)
        need(x.ndim == 3 and x.shape[1:] == (DESIGN_LEN, 4), 'Native input shape changed')
        need(np.all((x == 0) | (x == 1)) and np.all(x.sum(axis=-1) == 1), 'Invalid native one-hot')
        tokens = x.argmax(axis=-1).astype(np.uint8)
        self.last_tokens = tokens
        self.calls.append(len(tokens))
        p = self.scorer.predict(insert(self.contexts, tokens))
        self.last_predictions = p.reshape(len(tokens), len(self.contexts), -1).copy()
        if self.progress_fn and len(tokens) == N_SINGLE_MUTANTS:
            self.progress_fn(sum(n == N_SINGLE_MUTANTS for n in self.calls))
        return p.reshape(len(tokens), -1)


def native_api():
    from crested.tl.design import in_silico_evolution, EnhancerOptimizer
    need(all(k in inspect.signature(in_silico_evolution).parameters for k in
             ['n_mutations', 'model', 'enhancer_optimizer', 'starting_sequences', 'return_intermediate']),
         'Installed CREsted native ISE API differs; refusing custom fallback')
    return in_silico_evolution, EnhancerOptimizer


def evolve(scorer, contexts, initial, floor, steps=200, progress_fn=None, capture_replay=False):
    ise, optimizer = native_api()
    adapter = ContextAdapter(scorer, contexts, progress_fn)
    c = len(contexts); k = len(scorer.names)
    selections = []; replay_frames = []

    def choose(mutated_predictions, original_prediction, target, **kwargs):
        p = np.asarray(mutated_predictions).reshape(-1, c, k)
        need(len(p) == N_SINGLE_MUTANTS and adapter.last_tokens.shape == (N_SINGLE_MUTANTS, DESIGN_LEN),
             'Native ISE did not evaluate every single-base substitution')
        need(np.array_equal(p, adapter.last_predictions), 'Native optimizer prediction axes changed')
        best = int(np.argmax(objective(p, scorer.target, floor)))
        # The native API can call the optimizer again after the final iteration.
        # Keep the callback pure: only inspect the selected native variant.
        selections.append((adapter.last_tokens[best].copy(), p[best].copy()))
        if capture_replay:
            full = insert(contexts, adapter.last_tokens)
            frames = []; positions = []
            for j in range(c):
                row = best*c+j; start = row//scorer.batch_size*scorer.batch_size
                frame = full[start:start+scorer.batch_size].copy()
                if len(frame) < scorer.batch_size:
                    frame = np.concatenate([frame, np.repeat(frame[-1:], scorer.batch_size-len(frame), axis=0)])
                frames.append(frame); positions.append(row-start)
            replay_frames.append((np.asarray(frames), np.asarray(positions)))
        return best

    info, end = ise(n_mutations=steps, target=scorer.target, model=adapter,
                    n_sequences=1, starting_sequences=[dna(initial)],
                    return_intermediate=True, enhancer_optimizer=optimizer(optimize_func=choose))
    need(len(info) == len(end) == 1, 'Native sequence count differs')
    info = info[0]; trajectory = [initial.copy()]
    for pos, base in info['changes']:
        if pos == -1:
            continue
        seq = trajectory[-1].copy(); seq[int(pos)] = encode(str(base))[0]
        need(int((seq != trajectory[-1]).sum()) == 1, 'Native step is not exactly one substitution')
        trajectory.append(seq)
    trajectory = np.asarray(trajectory)
    predictions = np.asarray(info['predictions']).reshape(-1, c, k)
    need(trajectory.shape == (steps+1, DESIGN_LEN) and predictions.shape == (steps+1, c, k), 'Native history incomplete')
    need(np.array_equal(trajectory[-1], encode(str(end[0]))), 'Native endpoint/history mismatch')
    need(adapter.calls.count(N_SINGLE_MUTANTS) == steps, 'Native exhaustive iteration count differs')
    need(len(selections) in (steps, steps+1), 'Native optimizer callback count changed')
    for j, (tokens, prediction) in enumerate(selections[:steps], 1):
        need(np.array_equal(tokens, trajectory[j]), 'Native sequence/history mapping differs')
        need(np.array_equal(prediction, predictions[j]), 'Native selected score/history mapping differs')
    if len(selections) == steps+1:
        need(all(np.array_equal(a, b) for a, b in zip(selections[-1], selections[-2])),
             'Native final optimizer callback changed the endpoint')
    selected, feasible = best_step(predictions, scorer.target, floor)
    result = dict(trajectory=trajectory, predictions=predictions, floor=floor,
                selected_step=np.array(selected), feasible=np.array(feasible),
                mutation_batches=np.array(adapter.calls), target_index=np.array(scorer.target))
    if capture_replay:
        result['replay_frames'] = np.asarray([x[0] for x in replay_frames[:steps]])
        result['replay_positions'] = np.asarray([x[1] for x in replay_frames[:steps]])
    return result
