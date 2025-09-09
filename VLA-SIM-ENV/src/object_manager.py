from typing import Dict, Any, List
from src.utils.logger import logger

class ObjectManager:
    def __init__(self):
        self.objects: List[Dict[str, Any]] = []

    def add_objects(self, objs: List[Dict[str, Any]]):
        for o in objs:
            self.add_object(o)

    def add_object(self, obj: Dict[str, Any]):
        # Basic validation
        name = obj.get("name", f"obj_{len(self.objects)}")
        asset = obj.get("asset")
        if not asset:
            logger.warning(f"Object {name} missing asset path.")
        self.objects.append(obj)
        logger.info(f"Registered object: {name} ({asset})")
