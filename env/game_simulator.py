"""Combat game simulator: load a snapshot from JSON and wrap it in :class:`~environment.BalatroEnv`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, NamedTuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "balatro_lite_gym"))

from engine import GameSnapshot  # noqa: E402
from env.debug import print_snapshot  # noqa: E402
from env.snapshot_io import load_snapshot  # noqa: E402
from environment import (  # noqa: E402
    BalatroEnv,
    _is_invalid_selection,
    _selected_indices,
)

_DEFAULT_SNAPSHOT_JSON = _REPO / "temp" / "snapshot_no_jokers_level1.json"


class StepOutcome(NamedTuple):
    """Result of :meth:`GameSimulator.step_and_print`."""

    stepped: bool
    """``True`` if :meth:`~environment.BalatroEnv.step` ran."""
    terminated: bool = False
    """``True`` if the episode ended on that step."""
    combat_won: bool | None = None
    """On a terminal step: ``True`` = won, ``False`` = loss; else ``None``."""
    reward: float | None = None
    """Env step reward when ``stepped``; else ``None``."""


def _hint_if_action_invalid(action: Any, snap: GameSnapshot) -> str | None:
    """Return a short hint if *action* would fail or no-op like :meth:`BalatroEnv.step`; else ``None``."""
    if not isinstance(action, dict):
        return "action should be a dict with keys 'selection' and 'action_type'."
    if "selection" not in action or "action_type" not in action:
        return "action needs 'selection' (per-slot 0/1) and 'action_type' (0=discard, 1=play)."
    hand = snap.hand
    try:
        indices = _selected_indices(action["selection"], len(hand))
    except (TypeError, IndexError, ValueError):
        return "selection must be indexable; use 0/1 for each hand slot index 0..len(hand)-1."

    if _is_invalid_selection(indices):
        return "pick 1–5 cards from the hand (at least one slot, at most five)."

    at = action["action_type"]
    if at not in (0, 1):
        return "action_type must be 0 (discard) or 1 (play)."

    if at == 1 and snap.play_remaining <= 0:
        return "cannot play: no plays remaining (episode may be over)."
    if at == 0 and snap.discard_remaining <= 0:
        return "cannot discard: no discards remaining."

    return None


def _format_action_for_print(action: Any, snap: GameSnapshot) -> str:
    """One-line summary of *action* for logging (may be invalid)."""
    if not isinstance(action, dict):
        return f"action: {action!r}"
    if "selection" not in action or "action_type" not in action:
        return f"action: {action!r}"
    at = action["action_type"]
    if at == 1:
        verb = "play"
    elif at == 0:
        verb = "discard"
    else:
        verb = f"type={at!r}"
    try:
        idx = _selected_indices(action["selection"], len(snap.hand))
    except (TypeError, IndexError, ValueError):
        idx = "(selection unreadable)"
    return f"action: {verb}  hand_slots={idx}"


class GameSimulator:
    """Snapshot plus a Gymnasium env reset to that layout (``info['snapshot']`` is live state)."""

    __slots__ = ("snapshot", "env", "obs", "info")

    def __init__(
        self,
        snapshot: GameSnapshot,
        *,
        seed: int | None = 0,
    ) -> None:
        self.snapshot = snapshot
        self.env = BalatroEnv(snapshot)
        self.obs, self.info = self.env.reset(
            seed=seed,
            options={"snapshot": snapshot},
        )

    def print_info_snapshot(
        self,
        *,
        deck: Literal["summary", "full"] = "summary",
    ) -> None:
        """Pretty-print :attr:`info` ``['snapshot']`` (same object as live env state)."""
        print_snapshot(self.info["snapshot"], deck=deck)

    def step_and_print(
        self,
        action: dict,
        *,
        deck: Literal["summary", "full"] = "summary",
    ) -> StepOutcome:
        """If *action* is valid for the current :attr:`info` ``['snapshot']``, :meth:`step`, then print state.

        Invalid actions print a ``(hint) …`` line and do **not** call :meth:`~environment.BalatroEnv.step`.

        Always prints **pre-step** snapshot and the action line first; after a valid step, prints
        **post-step** snapshot as before.
        """
        snap = self.info["snapshot"]
        self.print_info_snapshot(deck=deck)
        print(_format_action_for_print(action, snap))
        hint = _hint_if_action_invalid(action, snap)
        if hint is not None:
            print(f"(hint) {hint}")
            return StepOutcome(False)
        self.obs, reward, terminated, _truncated, self.info = self.env.step(action)
        self.print_info_snapshot(deck=deck)
        cw = self.info.get("combat_won") if terminated else None
        return StepOutcome(True, bool(terminated), cw, float(reward))

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        seed: int | None = 0,
    ) -> GameSimulator:
        return cls(load_snapshot(path), seed=seed)


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
        default=0,
        help="Episode RNG seed (draw order); pass a negative value for None → random.",
    )
    args = p.parse_args()
    path = args.snapshot.resolve()
    if not path.is_file():
        sys.exit(f"snapshot not found: {path}")
    seed: int | None = None if args.seed < 0 else args.seed
    sim = GameSimulator.from_json(path, seed=seed)
    sim.print_info_snapshot(deck="summary")


if __name__ == "__main__":
    main()
