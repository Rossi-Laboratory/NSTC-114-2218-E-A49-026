import argparse
from typing import Dict, Any
from src.scene_loader import load_scene
from src.utils.logger import logger, banner, kv
from src.utils.visualizer import summarize_scene
from src.utils.metrics import stability_proxy, collision_proxy

def try_import_omniverse():
    try:
        import omni  # type: ignore
        import carb  # type: ignore
        return True
    except Exception:
        return False

class OmniverseEnv:
    def __init__(self, cfg: Dict[str, Any], dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run or (not try_import_omniverse())
        self.backend = "omniverse" if not self.dry_run else "mock"
        if self.dry_run:
            logger.warning("Omniverse not available or --dry-run set. Using MockSim backend.")
        else:
            logger.info("Omniverse detected. Initializing real simulator...")
            # TODO: Initialize Isaac Sim / USD stage here

    def load(self, scene: Dict[str, Any]):
        if self.backend == "mock":
            logger.info(f"[MockSim] load scene '{scene.get('name')}'")
        else:
            logger.info(f"[Omniverse] load scene '{scene.get('name')}' (USD creation omitted)")

    def step(self, n: int = 60):
        logger.info(f"[{self.backend}] stepping for {n} frames")

    def close(self):
        logger.info(f"[{self.backend}] closing")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--scene', required=True, type=str)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    banner("Omniverse Bootstrap")
    kv("Headless", cfg.get("headless", True))

    scene = load_scene(args.scene)
    summarize_scene(scene)

    env = OmniverseEnv(cfg, dry_run=args.dry_run)
    env.load(scene)
    env.step(120)

    kv("StabilityProxy", f"{stability_proxy(scene):.3f}")
    kv("CollisionProxy", collision_proxy())
    env.close()

if __name__ == "__main__":
    main()
