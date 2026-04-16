import sys
from pathlib import Path

# Project root (parent of tests/) on sys.path for `import engine` / `import environment`.
_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TESTS))
