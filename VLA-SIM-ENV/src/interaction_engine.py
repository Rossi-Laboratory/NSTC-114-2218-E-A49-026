from typing import Dict, Any, Optional, Tuple
from src.utils.logger import logger

class InteractionEngine:
    def __init__(self, backend: str = "mock"):
        self.backend = backend

    def pick(self, obj_name: str) -> bool:
        logger.info(f"[{self.backend}] PICK -> {obj_name}")
        return True

    def place(self, obj_name: str, target_pose: Tuple[float, float, float, float, float, float, float]) -> bool:
        logger.info(f"[{self.backend}] PLACE -> {obj_name} @ {target_pose}")
        return True

    def push(self, obj_name: str, direction_xy: Tuple[float, float], force: float = 1.0) -> bool:
        logger.info(f"[{self.backend}] PUSH -> {obj_name}, dir={direction_xy}, F={force}")
        return True

    def navigate(self, goal_pose: Tuple[float, float, float, float, float, float, float]) -> bool:
        logger.info(f"[{self.backend}] NAVIGATE -> {goal_pose}")
        return True

    def dispatch(self, intent: Dict[str, Any]) -> bool:
        # intent example: {"action": "pick_and_place", "object": "box", "target_pose": [...]}
        action = intent.get("action")
        if action == "pick":
            return self.pick(intent["object"])
        if action == "place":
            return self.place(intent["object"], tuple(intent["target_pose"]))
        if action == "push":
            return self.push(intent["object"], tuple(intent["dir"]), float(intent.get("force", 1.0)))
        if action == "navigate":
            return self.navigate(tuple(intent["target_pose"]))
        if action == "pick_and_place":
            ok1 = self.pick(intent["object"])
            ok2 = self.place(intent["object"], tuple(intent["target_pose"]))
            return ok1 and ok2
        logger.warning(f"Unknown action: {action}")
        return False
