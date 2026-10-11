import sys
from pathlib import Path

backend_dir = Path(__file__).parent.resolve()
engine_module_dir = backend_dir / "Engine_module"
engine_src_dir = engine_module_dir / "src"

for p in [str(backend_dir), str(engine_module_dir), str(engine_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
