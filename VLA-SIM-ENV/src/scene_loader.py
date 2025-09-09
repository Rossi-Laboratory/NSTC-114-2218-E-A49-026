import yaml
from typing import Dict, Any
from src.utils.logger import logger

REQUIRED_KEYS = ["name", "floor"]

def load_scene(scene_path: str) -> Dict[str, Any]:
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = yaml.safe_load(f)
    for k in REQUIRED_KEYS:
        if k not in scene:
            raise ValueError(f"Scene file missing required key: {k}")
    logger.info(f"Loaded scene: {scene.get('name')} from {scene_path}")
    return scene
