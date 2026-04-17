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
    hand_levels_by_type = {
        int(hand_type): random.randint(1, HAND_LEVEL_MAX) for hand_type in HandType
    }
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
        hand_levels=hand_levels_by_type,
    )


def generate_snapshots(count: int) -> List[GameSnapshot]:
    """Return ``count`` independently generated snapshots."""
    return [generate_snapshot() for _ in range(count)]


# ---------------------------------------------------------------------------
# JSON ↔ GameSnapshot (schema = temp/snapshot_no_jokers.json)
# ---------------------------------------------------------------------------

_REQUIRED_ROOT_JSON_KEYS = frozenset(
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


def _card_json(card: Card) -> dict:
    return {"card_id": card.card_id, "enhancement": card.enhancement, "edition": card.edition}


def _joker_json(joker: Joker) -> dict:
    return {"id": joker.id, "edition": joker.edition}


def _card_from_json(
    item: object, *, field_name: str, index_in_array: int
) -> Card:
    if not isinstance(item, dict):
        raise TypeError(
            f"{field_name}[{index_in_array}]: expected object, got {type(item).__name__}"
        )
    return Card(
        int(item["card_id"]),
        int(item["enhancement"]),
        int(item["edition"]),
    )


def _joker_from_json(
    item: object, *, field_name: str, index_in_array: int
) -> Joker:
    if not isinstance(item, dict):
        raise TypeError(
            f"{field_name}[{index_in_array}]: expected object, got {type(item).__name__}"
        )
    return Joker(int(item["id"]), int(item["edition"]))


def _cards_from_json(value: object, field_name: str) -> List[Card]:
    if not isinstance(value, list):
        raise TypeError(
            f"{field_name} must be a JSON array, got {type(value).__name__}"
        )
    return [
        _card_from_json(element, field_name=field_name, index_in_array=i)
        for i, element in enumerate(value)
    ]


def _jokers_from_json(value: object, field_name: str) -> List[Joker]:
    if not isinstance(value, list):
        raise TypeError(
            f"{field_name} must be a JSON array, got {type(value).__name__}"
        )
    return [
        _joker_from_json(element, field_name=field_name, index_in_array=i)
        for i, element in enumerate(value)
    ]


def _hand_levels_from_json(hand_levels_value: object) -> dict[int, int]:
    if not isinstance(hand_levels_value, dict):
        raise TypeError(
            "hand_levels must be a JSON object, got "
            f"{type(hand_levels_value).__name__}"
        )
    levels_by_hand_type: dict[int, int] = {}
    for hand_type_key, level_value in hand_levels_value.items():
        if not isinstance(level_value, (int, float)) or isinstance(level_value, bool):
            raise TypeError(
                f"hand_levels[{hand_type_key!r}]: expected numeric level, "
                f"got {type(level_value).__name__}"
            )
        level = int(level_value)
        if level < 1:
            raise ValueError(
                f"hand_levels[{hand_type_key!r}]: level must be >= 1, got {level}"
            )
        levels_by_hand_type[int(hand_type_key)] = level
    return levels_by_hand_type


def snapshot_to_dict(snapshot: GameSnapshot) -> dict:
    """Serialize to the canonical snapshot JSON object (key order matches the reference file)."""
    return {
        "target_score": snapshot.target_score,
        "current_score": snapshot.current_score,
        "blind_id": snapshot.blind_id,
        "hand": [_card_json(card) for card in snapshot.hand],
        "deck": [_card_json(card) for card in snapshot.deck],
        "jokers": [_joker_json(joker) for joker in snapshot.jokers],
        "play_remaining": snapshot.play_remaining,
        "discard_remaining": snapshot.discard_remaining,
        "player_hand_size": snapshot.player_hand_size,
        "hand_levels": {
            str(hand_type): int(level)
            for hand_type, level in snapshot.hand_levels.items()
        },
    }


def dict_to_snapshot(root: dict) -> GameSnapshot:
    """Parse the canonical snapshot JSON object into a ``GameSnapshot``."""
    missing_keys = _REQUIRED_ROOT_JSON_KEYS - root.keys()
    if missing_keys:
        raise KeyError(
            "snapshot JSON missing keys: " + ", ".join(sorted(missing_keys))
        )
    return GameSnapshot(
        target_score=int(root["target_score"]),
        current_score=int(root["current_score"]),
        blind_id=int(root["blind_id"]),
        hand=_cards_from_json(root["hand"], "hand"),
        deck=_cards_from_json(root["deck"], "deck"),
        jokers=_jokers_from_json(root["jokers"], "jokers"),
        play_remaining=int(root["play_remaining"]),
        discard_remaining=int(root["discard_remaining"]),
        player_hand_size=int(root["player_hand_size"]),
        hand_levels=_hand_levels_from_json(root["hand_levels"]),
    )


def save_snapshot(path: Union[str, Path], snapshot: GameSnapshot) -> None:
    """Write one canonical JSON object per file (readable by :func:`load_snapshot`)."""
    with open(path, "w", encoding="utf-8") as out_file:
        json.dump(snapshot_to_dict(snapshot), out_file, indent=2, ensure_ascii=False)


def load_snapshot(path: Union[str, Path]) -> GameSnapshot:
    """Load one canonical JSON object from a file (not a JSON array)."""
    snapshot_path = Path(path)
    with open(snapshot_path, "r", encoding="utf-8") as in_file:
        root = json.load(in_file)
    if not isinstance(root, dict):
        raise TypeError(
            f"expected one JSON object per file in {snapshot_path!s}, "
            f"got {type(root).__name__}"
        )
    return dict_to_snapshot(root)


def load_snapshot_pool_from_json_dir(
    directory: Union[str, Path],
    *,
    pattern: str = "*.json",
    sort: bool = True,
) -> List[GameSnapshot]:
    """Load each matching file under ``directory`` with :func:`load_snapshot`."""
    root_dir = Path(directory)
    if not root_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {root_dir}")
    paths = sorted(root_dir.glob(pattern)) if sort else list(root_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root_dir}")
    return [load_snapshot(path) for path in paths]


if __name__ == "__main__":
    snapshots = generate_snapshots(3)
    out_dir = Path("snapshots_out")
    out_dir.mkdir(exist_ok=True)
    for i, s in enumerate(snapshots):
        save_snapshot(out_dir / f"{i}.json", s)
    print(f"Saved {len(snapshots)} snapshots under {out_dir}/")

    loaded0 = load_snapshot(out_dir / "0.json")
    print(f"Loaded snapshot 0: target_score={loaded0.target_score}")
