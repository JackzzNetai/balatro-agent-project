"""Load and save :class:`~engine.GameSnapshot` as a single JSON object per file.

Canonical on-disk layout matches ``temp/snapshot_no_jokers.json`` — one root object
with these keys (in this order when writing):

- ``target_score``, ``current_score``, ``blind_id``
- ``hand``, ``deck``: arrays of ``{ "card_id", "enhancement", "edition" }``
- ``jokers``: array of ``{ "id", "edition" }``
- ``play_remaining``, ``discard_remaining``, ``player_hand_size``
- ``hand_levels``: object mapping hand-type id (string) to level (integer ``>= 1``)

Also includes optional :func:`generate_snapshot` / :func:`generate_snapshots` for demos.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Union

from defs import HandType
from engine import Card, GameSnapshot, Joker

# ---------------------------------------------------------------------------
# Demo snapshot generation (not used by load/save)
# ---------------------------------------------------------------------------

TARGET_SCORE_MIN = 300
TARGET_SCORE_MAX = 10000

PLAY_REMAINING_MAX = 5
DISCARD_REMAINING_MAX = 4

PLAYER_HAND_SIZE_MIN = 5
PLAYER_HAND_SIZE_MAX = 12

HAND_LEVEL_MAX = 20


def generate_snapshot() -> GameSnapshot:
    """Return a random ``GameSnapshot`` within the configured bounds."""
    hand_levels = {int(h): random.randint(1, HAND_LEVEL_MAX) for h in HandType}
    return GameSnapshot(
        target_score=random.randint(TARGET_SCORE_MIN, TARGET_SCORE_MAX),
        current_score=0,
        blind_id=random.randint(0, 7),
        hand=[],
        deck=[],
        jokers=[],
        play_remaining=random.randint(1, PLAY_REMAINING_MAX),
        discard_remaining=random.randint(0, DISCARD_REMAINING_MAX),
        player_hand_size=random.randint(PLAYER_HAND_SIZE_MIN, PLAYER_HAND_SIZE_MAX),
        hand_levels=hand_levels,
    )


def generate_snapshots(n: int) -> List[GameSnapshot]:
    """Return *n* independently generated snapshots."""
    return [generate_snapshot() for _ in range(n)]


# ---------------------------------------------------------------------------
# JSON ↔ GameSnapshot (schema = temp/snapshot_no_jokers.json)
# ---------------------------------------------------------------------------

_SNAPSHOT_KEYS = frozenset(
    {
        "target_score",
        "current_score",
        "blind_id",
        "hand",
        "deck",
        "jokers",
        "play_remaining",
        "discard_remaining",
        "player_hand_size",
        "hand_levels",
    }
)


def _card_json(c: Card) -> dict:
    return {"card_id": c.card_id, "enhancement": c.enhancement, "edition": c.edition}


def _joker_json(j: Joker) -> dict:
    return {"id": j.id, "edition": j.edition}


def _card_from_json(x: object, *, where: str, index: int) -> Card:
    if not isinstance(x, dict):
        raise TypeError(f"{where}[{index}]: expected object, got {type(x).__name__}")
    return Card(
        int(x["card_id"]),
        int(x["enhancement"]),
        int(x["edition"]),
    )


def _joker_from_json(x: object, *, where: str, index: int) -> Joker:
    if not isinstance(x, dict):
        raise TypeError(f"{where}[{index}]: expected object, got {type(x).__name__}")
    return Joker(int(x["id"]), int(x["edition"]))


def _cards_from_json(xs: object, field: str) -> List[Card]:
    if not isinstance(xs, list):
        raise TypeError(f"{field} must be a JSON array, got {type(xs).__name__}")
    return [_card_from_json(x, where=field, index=i) for i, x in enumerate(xs)]


def _jokers_from_json(xs: object, field: str) -> List[Joker]:
    if not isinstance(xs, list):
        raise TypeError(f"{field} must be a JSON array, got {type(xs).__name__}")
    return [_joker_from_json(x, where=field, index=i) for i, x in enumerate(xs)]


def _hand_levels_from_json(raw: object) -> dict[int, int]:
    if not isinstance(raw, dict):
        raise TypeError(f"hand_levels must be a JSON object, got {type(raw).__name__}")
    out: dict[int, int] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"hand_levels[{key!r}]: expected numeric level, got {type(value).__name__}"
            )
        level = int(value)
        if level < 1:
            raise ValueError(f"hand_levels[{key!r}]: level must be >= 1, got {level}")
        out[int(key)] = level
    return out


def snapshot_to_dict(snapshot: GameSnapshot) -> dict:
    """Serialize to the canonical snapshot JSON object (key order matches the reference file)."""
    return {
        "target_score": snapshot.target_score,
        "current_score": snapshot.current_score,
        "blind_id": snapshot.blind_id,
        "hand": [_card_json(c) for c in snapshot.hand],
        "deck": [_card_json(c) for c in snapshot.deck],
        "jokers": [_joker_json(j) for j in snapshot.jokers],
        "play_remaining": snapshot.play_remaining,
        "discard_remaining": snapshot.discard_remaining,
        "player_hand_size": snapshot.player_hand_size,
        "hand_levels": {str(k): int(v) for k, v in snapshot.hand_levels.items()},
    }


def dict_to_snapshot(d: dict) -> GameSnapshot:
    """Parse the canonical snapshot JSON object into a ``GameSnapshot``."""
    missing = _SNAPSHOT_KEYS - d.keys()
    if missing:
        raise KeyError(f"snapshot JSON missing keys: {', '.join(sorted(missing))}")
    return GameSnapshot(
        target_score=int(d["target_score"]),
        current_score=int(d["current_score"]),
        blind_id=int(d["blind_id"]),
        hand=_cards_from_json(d["hand"], "hand"),
        deck=_cards_from_json(d["deck"], "deck"),
        jokers=_jokers_from_json(d["jokers"], "jokers"),
        play_remaining=int(d["play_remaining"]),
        discard_remaining=int(d["discard_remaining"]),
        player_hand_size=int(d["player_hand_size"]),
        hand_levels=_hand_levels_from_json(d["hand_levels"]),
    )


def save_snapshot(path: Union[str, Path], snapshot: GameSnapshot) -> None:
    """Write one canonical JSON object per file (readable by :func:`load_snapshot`)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot_to_dict(snapshot), f, indent=2, ensure_ascii=False)


def load_snapshot(path: Union[str, Path]) -> GameSnapshot:
    """Load one canonical JSON object from a file (not a JSON array)."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(
            f"expected one JSON object per file in {p!s}, got {type(data).__name__}"
        )
    return dict_to_snapshot(data)


def load_snapshot_pool_from_json_dir(
    directory: Union[str, Path],
    *,
    pattern: str = "*.json",
    sort: bool = True,
) -> List[GameSnapshot]:
    """Load each matching file under ``directory`` with :func:`load_snapshot`."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    paths = sorted(d.glob(pattern)) if sort else list(d.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {d}")
    return [load_snapshot(p) for p in paths]
