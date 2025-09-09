from src.scene_loader import load_scene
from src.utils.visualizer import summarize_scene
from src.utils.logger import banner, kv
from src.init_genesis import GenesisEnv
import yaml

def main():
    cfg = yaml.safe_load(open("configs/genesis_config.yaml", "r", encoding="utf-8"))
    scene = load_scene("configs/scenes/household.yaml")
    banner("Single Robot Demo")
    summarize_scene(scene)
    env = GenesisEnv(cfg, dry_run=True)
    env.load(scene)
    env.step(60)
    env.close()

if __name__ == "__main__":
    main()
