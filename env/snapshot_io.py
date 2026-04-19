"""Load and save :class:`~engine.GameSnapshot` as a single JSON object per file.

Canonical on-disk layout matches ``temp/snapshot_no_jokers.json`` — one root object
with these keys (in this order when writing):

- ``target_score``, ``current_score``, ``blind_id``
- ``hand``, ``deck``: arrays of ``{ "card_id", "enhancement", "edition" }``
- ``jokers``: array of ``{ "id", "edition" }``
- ``play_remaining``, ``discard_remaining``, ``player_hand_size``
- ``hand_levels``: object mapping hand-type id (string) to level (integer ``>= 1``)

:func:`generate_snapshot` builds a full, load-compatible demo snapshot (see module
constants for target score, plays/discards, hand size). Optional
:class:`SnapshotGenerateOption` forces the 8-card hand to contain five cards that form
a flush (:attr:`SnapshotGenerateOption.FLUSH_IN_HAND`) or a straight
(:attr:`SnapshotGenerateOption.STRAIGHT_IN_HAND`); the rest of the deal is unchanged
(full 52-card partition, plain cards, same ``hand_levels`` distribution).
:func:`generate_snapshots` repeats generation with the same ``option``.
"""

from __future__ import annotations

import json
import random
import sys
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Union

# Running `python env/snapshot_io.py` (or "Run Python File") without PYTHONPATH:
# put the repo's balatro_lite_gym on sys.path so `defs` / `engine` import.
_gym_root = Path(__file__).resolve().parent.parent / "balatro_lite_gym"
if _gym_root.is_dir():
    _gym_root_s = str(_gym_root)
    if _gym_root_s not in sys.path:
        sys.path.insert(0, _gym_root_s)

from defs import CardEnhancement, Edition, HandType, NO_BOSS_BLIND_ID, NUM_RANKS
from engine import Card, GameSnapshot, Joker


class SnapshotGenerateOption(Enum):
    """How to pick the five \"feature\" cards when building a structured demo hand."""

    FLUSH_IN_HAND = auto()
    STRAIGHT_IN_HAND = auto()


# ---------------------------------------------------------------------------
# Demo snapshot generation (not used by load/save)
# ---------------------------------------------------------------------------

TARGET_SCORE = 500
PLAY_REMAINING = 4
DISCARD_REMAINING = 4
PLAYER_HAND_SIZE = 8
FULL_DECK_SIZE = 52
DEMO_SNAPSHOT_COUNT = 25

_PLAIN_ENHANCEMENT = int(CardEnhancement.NONE)
_PLAIN_EDITION = int(Edition.BASE)


def _plain_card(card_id: int) -> Card:
    return Card(card_id, _PLAIN_ENHANCEMENT, _PLAIN_EDITION)


def _five_card_ids(option: SnapshotGenerateOption) -> List[int]:
    if option is SnapshotGenerateOption.FLUSH_IN_HAND:
        suit = random.randrange(4)
        ranks = random.sample(range(NUM_RANKS), 5)
        return [suit * NUM_RANKS + r for r in ranks]
    if option is SnapshotGenerateOption.STRAIGHT_IN_HAND:
        start = random.randrange(0, NUM_RANKS - 3)
        if start < 9:
            ranks = list(range(start, start + 5))
        else:
            ranks = [9, 10, 11, 12, 0]
        return [random.randrange(4) * NUM_RANKS + r for r in ranks]
    raise ValueError(f"unknown snapshot generate option: {option!r}")


def _hand_deck_from_feature_ids(five_ids: List[int]) -> Tuple[List[Card], List[Card]]:
    pool = set(range(FULL_DECK_SIZE)) - set(five_ids)
    extra = random.sample(sorted(pool), PLAYER_HAND_SIZE - 5)
    hand_ids = list(five_ids) + extra
    random.shuffle(hand_ids)
    deck_ids = list(pool - set(extra))
    random.shuffle(deck_ids)
    return [_plain_card(i) for i in hand_ids], [_plain_card(i) for i in deck_ids]


def _random_hand_levels() -> dict[int, int]:
    return {
        int(hand_type): random.choices([1, 2], weights=[9, 1], k=1)[0]
        for hand_type in HandType
    }


def generate_snapshot(
    *, option: SnapshotGenerateOption | None = None
) -> GameSnapshot:
    """Return one full ``GameSnapshot`` compatible with :func:`dict_to_snapshot` / JSON I/O.

    With ``option is None``, shuffles card ids ``0 .. FULL_DECK_SIZE - 1``, deals
    ``PLAYER_HAND_SIZE`` into ``hand``, remainder into ``deck``.

    With ``option`` set to :class:`SnapshotGenerateOption`, builds a hand that **contains**
    five cards forming that poker shape (flush or straight); the other three hand cards
    and the 44-card draw pile are random subject to a full-deck partition. Hand card
    order is shuffled.

    Fixed combat fields: ``TARGET_SCORE``, ``PLAY_REMAINING``, ``DISCARD_REMAINING``,
    ``NO_BOSS_BLIND_ID``, empty jokers, plain cards only. Each ``HandType`` level is 1
    with probability 0.9 else 2 (independent).
    """
    if option is None:
        ids = list(range(FULL_DECK_SIZE))
        random.shuffle(ids)
        hand = [_plain_card(cid) for cid in ids[:PLAYER_HAND_SIZE]]
        deck = [_plain_card(cid) for cid in ids[PLAYER_HAND_SIZE:]]
    else:
        hand, deck = _hand_deck_from_feature_ids(_five_card_ids(option))
    hand_levels_by_type = _random_hand_levels()
    return GameSnapshot(
        target_score=TARGET_SCORE,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=hand,
        deck=deck,
        jokers=[],
        play_remaining=PLAY_REMAINING,
        discard_remaining=DISCARD_REMAINING,
        player_hand_size=PLAYER_HAND_SIZE,
        hand_levels=hand_levels_by_type,
    )


def generate_snapshots(
    count: int, *, option: SnapshotGenerateOption | None = None
) -> List[GameSnapshot]:
    """Return ``count`` independent :func:`generate_snapshot` results (one JSON object each when saved)."""
    return [generate_snapshot(option=option) for _ in range(count)]


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
    option: SnapshotGenerateOption | None = SnapshotGenerateOption.FLUSH_IN_HAND
    snapshots = generate_snapshots(DEMO_SNAPSHOT_COUNT, option=option)
    name_suffix = f"{option.name.lower()}_" if option is not None else ""
    out_dir = Path("temp/snapshots_out")
    out_dir.mkdir(exist_ok=True)
    for i, s in enumerate(snapshots):
        save_snapshot(out_dir / f"{name_suffix}{i}.json", s)
    print(f"Saved {len(snapshots)} snapshots under {out_dir}/")
