# acp/models/fast/aggregator.py
import torch

def segment_mean(x, ends_bool):
    B, T, D = x.shape
    out = []
    maxN = 0
    for b in range(B):
        xb = x[b]; eb = ends_bool[b]
        idxs = torch.nonzero(eb, as_tuple=False).flatten().tolist()
        start=0; embs=[]
        for i in idxs:
            seg = xb[start:i+1]
            if seg.numel()>0: embs.append(seg.mean(0))
            start = i+1
        if start<T: embs.append(xb[start:].mean(0))
        if len(embs)==0: embs=[xb.mean(0)]
        E = torch.stack(embs,0)
        out.append(E); maxN=max(maxN,E.size(0))
    y = x.new_zeros(B, maxN, D)
    for b,E in enumerate(out): y[b,:E.size(0)] = E
    return y
