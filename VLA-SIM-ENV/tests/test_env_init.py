from src.scene_loader import load_scene

def test_load_scene():
    scene = load_scene("configs/scenes/multi_object.yaml")
    assert scene["name"] == "multi_object_arena"
    assert "objects" in scene
