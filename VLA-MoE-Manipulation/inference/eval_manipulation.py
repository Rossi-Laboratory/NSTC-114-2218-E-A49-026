# Placeholder for evaluation code.
# In practice, wire this to your benchmark (e.g., success@N for grasp/place/insert/rotate).
import numpy as np

def success_rate(pred_actions, gt_actions, thresh=0.25):
    diff = np.linalg.norm(pred_actions - gt_actions, axis=-1)
    return (diff < thresh).mean()
