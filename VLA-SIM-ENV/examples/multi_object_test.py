from src.scene_loader import load_scene
from src.utils.visualizer import summarize_scene
from src.utils.logger import banner
from src.init_omniverse import OmniverseEnv
import yaml

def main():
    cfg = yaml.safe_load(open("configs/omniverse_config.yaml", "r", encoding="utf-8"))
    scene = load_scene("configs/scenes/multi_object.yaml")
    banner("Multi-Object Test")
    summarize_scene(scene)
    env = OmniverseEnv(cfg, dry_run=True)
    env.load(scene)
    env.step(120)
    env.close()

if __name__ == "__main__":
    main()
