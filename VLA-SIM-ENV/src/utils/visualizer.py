from typing import Dict, Any, List
from rich.console import Console

_console = Console()

def summarize_scene(scene: Dict[str, Any]):
    _console.print(f"[bold]Scene[/bold]: {scene.get('name','(unnamed)')}")
    for sec in ("floor", "robots", "objects", "goals"):
        if sec in scene and scene[sec]:
            _console.print(f"  - {sec}:")
            items = scene[sec] if isinstance(scene[sec], list) else [scene[sec]]
            for it in items:
                _console.print(f"    • {it.get('name', it.get('type', 'item'))}")
