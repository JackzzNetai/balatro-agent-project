"""Build :class:`~environment.BalatroEnv` from a snapshot and step it (optional pretty-print)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "balatro_lite_gym"))

from engine import GameSnapshot  # noqa: E402
from env.debug import print_snapshot as _dump_snapshot  # noqa: E402
from env.snapshot_io import load_snapshot  # noqa: E402
from environment import BalatroEnv, _selected_indices  # noqa: E402

_DEFAULT_SNAPSHOT_JSON = _REPO / "temp" / "snapshot_no_jokers_level1.json"


def _action_line(action: Any, snap: GameSnapshot) -> str:
    if not isinstance(action, dict):
        return f"action: {action!r}"
    if "selection" not in action or "action_type" not in action:
        return f"action: {action!r}"
    at = action["action_type"]
    verb = (
        "play"
        if at == 1
        else "discard"
        if at == 0
        else f"type={at!r}"
    )
    try:
        idx = _selected_indices(action["selection"], len(snap.hand))
    except (TypeError, IndexError, ValueError):
        idx = "(bad selection)"
    return f"action: {verb}  hand_slots={idx}"


class GameSimulator:
    """Holds a :class:`~environment.BalatroEnv` reset to a layout; ``info['snapshot']`` is live state."""

    __slots__ = ("env", "obs", "info")

    def __init__(self, snapshot: GameSnapshot, *, seed: int | None = None) -> None:
        self.env = BalatroEnv(snapshot)
        self.obs, self.info = self.env.reset(
            seed=seed,
            options={"snapshot": snapshot},
        )

    @classmethod
    def from_json(cls, path: Path, *, seed: int | None = None) -> GameSimulator:
        return cls(load_snapshot(path), seed=seed)

    @property
    def snapshot(self) -> GameSnapshot:
        return self.info["snapshot"]

    def print_snapshot(self, *, deck: Literal["summary", "full"] = "summary") -> None:
        _dump_snapshot(self.snapshot, deck=deck)

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        """One Gymnasium step; updates :attr:`obs` and :attr:`info`."""
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        return self.obs, reward, terminated, truncated, self.info

    def step_print(
        self,
        action,
        *,
        deck: Literal["summary", "full"] = "summary",
    ) -> tuple[Any, float, bool, bool, dict]:
        """Print a one-line action summary, :meth:`step`, then dump the resulting snapshot."""
        print(_action_line(action, self.snapshot))
        out = self.step(action)
        self.print_snapshot(deck=deck)
        return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Load snapshot JSON, build BalatroEnv, reset to that layout."
    )
    p.add_argument(
        "snapshot",
        type=Path,
        nargs="?",
        default=_DEFAULT_SNAPSHOT_JSON,
        help=f"path to snapshot JSON (default: {_DEFAULT_SNAPSHOT_JSON.relative_to(_REPO)})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Episode RNG seed (draw order). Default: random. Pass a non-negative int for reproducibility.",
    )
    args = p.parse_args()
    path = args.snapshot.resolve()
    if not path.is_file():
        sys.exit(f"snapshot not found: {path}")
    seed: int | None = None if args.seed < 0 else args.seed
    sim = GameSimulator.from_json(path, seed=seed)
    sim.print_snapshot(deck="summary")


if __name__ == "__main__":
    main()
