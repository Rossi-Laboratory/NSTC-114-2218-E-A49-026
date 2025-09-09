from src.interaction_engine import InteractionEngine

def test_pick_place():
    eng = InteractionEngine(backend="mock")
    assert eng.pick("cube_A")
    assert eng.place("cube_A", (0,0,0,0,0,0,1))
