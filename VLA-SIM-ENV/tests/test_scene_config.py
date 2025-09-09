import yaml

def test_scene_keys():
    for fn in ["configs/scenes/multi_object.yaml", "configs/scenes/household.yaml", "configs/scenes/industrial.yaml"]:
        scene = yaml.safe_load(open(fn, "r", encoding="utf-8"))
        assert "name" in scene
        assert "floor" in scene
