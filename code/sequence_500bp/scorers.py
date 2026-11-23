import numpy as np
from design_core import onehot
from io_helpers import need, read, sha


def build_borzoi(outputs):
    # Exact architecture/head names from the completed matched_v1 experiment.
    # No pretrained download: the selected fine-tuned weights load strictly.
    import keras
    import crested.tl.zoo
    base=crested.tl.zoo.borzoi(seq_len=2048,target_length=64,num_classes=2608)
    x=keras.layers.Flatten(name='panel_flatten')(base.get_layer('final_conv_activation').output)
    y=keras.layers.Dense(outputs,activation='softplus',name='panel_output')(x)
    return keras.Model(base.inputs,y,name='Borzoi_target_panel')


class Scorer:
    def __init__(self,model,names,weights,device,batch_size=32,width=2048):
        self.model=model.to(device)
        self.names=list(names);self.target=self.names.index('TARGET_C1')
        self.weights=np.asarray(weights,dtype=np.float32)
        need(self.weights.shape==(len(names),) and np.isfinite(self.weights).all()
             and (self.weights>0).all(),'Invalid normalization weights')
        self.device=device;self.batch_size=batch_size;self.width=width
        need(model.input_shape[1:]==(width,4) and model.output_shape==(None,len(names)),'Model axes differ')
        for p in model.parameters():p.requires_grad_(False)
        model.eval()

    def _input(self,tokens):
        if self.width==500 and tokens.shape[1]==2048:tokens=tokens[:,774:1274]
        need(tokens.shape[1]==self.width,'Scorer input length differs')
        return tokens

    def predict(self,tokens):
        import torch
        tokens=self._input(np.asarray(tokens,dtype=np.uint8));out=[]
        with torch.no_grad():
            for start in range(0,len(tokens),self.batch_size):
                chunk=tokens[start:start+self.batch_size]
                count=len(chunk)
                # Fix inference batch shape so initial, mutant and resume scores
                # use the same GPU kernels. Padding is discarded from outputs.
                if count<self.batch_size:
                    chunk=np.concatenate([chunk,np.repeat(chunk[-1:],self.batch_size-count,axis=0)])
                x=torch.from_numpy(onehot(chunk)).to(self.device)
                p=(self.model(x,training=False)+self.model(torch.flip(x,[1,2]),training=False))/2
                p=p.detach().cpu().numpy()[:count]/self.weights[None,:]
                need(np.isfinite(p).all() and (p>=0).all(),'Nonfinite/negative model output')
                out.append(p)
        return np.concatenate(out)



def primary(settings,device):
    from pathlib import Path
    root=Path(settings['source'])
    meta=read(root/'data/atac_C1/DATASET.json')
    best=read(root/'models/atac_C1/BEST.json')
    need(sha(root/'data/atac_C1/DATASET.json')==settings['expected_primary_dataset'],'Primary metadata changed')
    need(best['sha256']==settings['expected_primary_weights']
         and sha(root/'models/atac_C1'/best['weights_file'])==settings['expected_primary_weights'],'Primary weights changed')
    model=build_borzoi(len(meta['names']))
    model.load_weights(root/'models/atac_C1'/best['weights_file'])
    return Scorer(model,meta['names'],meta['weights'],device,settings['batch_size'])


def secondary(settings,device):
    from pathlib import Path
    import crested
    root=Path(settings['secondary_source'])
    meta=read(root/'data/DATASET.json')
    need(sha(root/'data/DATASET.json')==settings['expected_secondary_dataset'],'Secondary metadata changed')
    need(sha(root/'base/BEST.keras')==settings['expected_secondary_weights'],'Secondary weights changed')
    model=crested.utils.load_model(root/'base/BEST.keras',compile=False)
    return Scorer(model,meta['classes'],meta['weights'],device,settings['batch_size'],width=500)
