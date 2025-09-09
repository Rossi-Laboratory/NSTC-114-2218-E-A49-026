from typing import Dict, Any
import random

def stability_proxy(scene: Dict[str, Any]) -> float:
    # Mock: proxy increases with heavy objects on floor and fewer randomizations
    objs = scene.get("objects", [])
    base = 0.7 + 0.05*sum(1 for o in objs if (o.get("physical",{}).get("mass",0) > 5))
    return min(1.0, max(0.0, base - 0.1*random.random()))

def collision_proxy() -> int:
    # Mock collisions draw
    return int(random.random() < 0.2)
