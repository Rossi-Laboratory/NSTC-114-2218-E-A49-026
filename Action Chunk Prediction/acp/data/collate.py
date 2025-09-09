# acp/data/collate.py
def collate_fn(batch):
    # simple collate; data are already tensors of equal length in this minimal example
    keys = batch[0].keys()
    out = {k: None for k in keys}
    for k in keys:
        out[k] = batch[0][k].new_tensor([b[k] for b in batch])
    return out
