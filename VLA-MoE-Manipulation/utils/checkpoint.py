import os, torch

def save_checkpoint(path, model, optimizer=None, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        **extra
    }
    torch.save(payload, path)
    print(f"[INFO] Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer=None):
    payload = torch.load(path, map_location='cpu')
    model.load_state_dict(payload['model'])
    if optimizer is not None and payload.get('optimizer'):
        optimizer.load_state_dict(payload['optimizer'])
    return payload
