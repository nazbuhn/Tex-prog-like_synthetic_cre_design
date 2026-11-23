import numpy as np
from io_helpers import need

ALPHABET=np.array(list('ACGTN'))


def dna(tokens):
    return ''.join(ALPHABET[np.asarray(tokens,dtype=int)])


def onehot(tokens):
    return np.concatenate([np.eye(4,dtype=np.float32),np.zeros((1,4),np.float32)])[tokens]


def insert(backbones,sequences):
    """Sequence-major batches; immutable flanks, exact central200bp replacement."""
    backbones=np.asarray(backbones,dtype=np.uint8);sequences=np.asarray(sequences,dtype=np.uint8)
    if sequences.ndim==1:sequences=sequences[None,:]
    need(sequences.shape[1]==200,'Insert must be200bp')
    n,c=len(sequences),len(backbones)
    out=np.broadcast_to(backbones,(n,)+backbones.shape).copy()
    start=(backbones.shape[1]-200)//2
    out[:,:,start:start+200]=sequences[:,None,:]
    return out.reshape(n*c,backbones.shape[1])


def dinuc_shuffle(tokens,rng):
    """Random Euler trail: preserves length, endpoints, mono/di-nucleotide counts.

    Not a uniform sample of every possible shuffle. Identical shuffles are
    possible and are counted explicitly in the control report.
    """
    tokens=np.asarray(tokens,dtype=np.uint8)
    need(tokens.ndim==1 and len(tokens)>1 and tokens.max()<4,'Shuffle requires ACGT')
    edges=[[] for _ in range(4)]
    for a,b in zip(tokens[:-1],tokens[1:]):edges[int(a)].append(int(b))
    for x in edges:rng.shuffle(x)
    stack=[int(tokens[0])];trail=[]
    while stack:
        u=stack[-1]
        if edges[u]:stack.append(edges[u].pop())
        else:trail.append(stack.pop())
    out=np.array(trail[::-1],np.uint8)
    need(len(out)==len(tokens),'Incomplete shuffle trail')
    return out


def margin(profiles,target):
    others=np.delete(np.arange(profiles.shape[-1]),target)
    return np.log1p(profiles[...,target])-np.log1p(profiles[...,others].max(axis=-1))


def robust(margins):
    return .5*np.mean(margins,axis=-1)+.5*np.min(margins,axis=-1)


def longest_run(tokens):
    a=np.asarray(tokens);bounds=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)]
    return int(np.diff(bounds).max())


def composition_ok(candidate,initial,gc_tolerance=.1,max_run=6):
    candidate=np.asarray(candidate);initial=np.asarray(initial)
    gc=lambda x:np.mean((x==1)|(x==2))
    return bool(candidate.max()<4 and abs(gc(candidate)-gc(initial))<=gc_tolerance+1e-12
                and longest_run(candidate)<=max(max_run,longest_run(initial)))



def shared_context_ids(rates,counts,splits,cells,excluded,n_contexts,rng):
    major=np.flatnonzero(np.asarray(cells)>=2000)
    need(len(major)>=2,'Too few adequately sampled context groups')
    train=np.flatnonzero(splits==0)
    rr=np.asarray(rates[np.ix_(train,major)])
    cc=np.asarray(counts[np.ix_(train,major)])
    keep=(cc>=3).all(axis=1)&(rr>0).all(axis=1)&~np.isin(train,excluded)
    ids=train[keep];rr=rr[keep]
    need(len(ids)>=n_contexts,'Too few training shared-accessibility backgrounds; no threshold relaxation')
    spread=np.log(rr.max(axis=1)/rr.min(axis=1))
    pool=ids[np.lexsort((ids,spread))[:min(1024,len(ids))]]
    return rng.choice(pool,n_contexts,replace=False)
