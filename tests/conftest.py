import sys
from pathlib import Path

# Repo root is the parent of this ``tests/`` directory.
_ROOT = Path(__file__).resolve().parents[1]
_GYM = _ROOT / "balatro_lite_gym"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_GYM))
