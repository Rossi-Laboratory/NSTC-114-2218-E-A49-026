from src.scene_loader import load_scene
from src.utils.visualizer import summarize_scene
from src.utils.logger import banner
from src.interaction_engine import InteractionEngine
import yaml

def main():
    scene = load_scene("configs/scenes/industrial.yaml")
    banner("VLA Integration (Stub)")
    summarize_scene(scene)

    engine = InteractionEngine(backend="mock")
    intents = [
        {"action": "pick", "object": "box"},
        {"action": "place", "object": "box", "target_pose": [0.2, 0.6, 0.15, 0, 0, 0, 1]},
        {"action": "navigate", "target_pose": [0.0, -0.2, 0.0, 0, 0, 0, 1]},
        {"action": "push", "object": "pallet", "dir": [1.0, 0.0], "force": 2.0},
    ]
    for intent in intents:
        engine.dispatch(intent)

if __name__ == "__main__":
    main()
