"""JSON serialization and file I/O for :class:`~engine.GameSnapshot`.

Also contains optional random snapshot generation for local demos (see
:func:`generate_snapshot`).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Union

from defs import HandType
from defs.poker_hands import POKER_HAND_BASE_CHIPS_MULT, POKER_HAND_LEVEL_UP_CHIPS_MULT
from engine import Card, GameSnapshot, Joker

# ---------------------------------------------------------------------------
# Configurable bounds — adjust these constants to control snapshot generation
# ---------------------------------------------------------------------------

TARGET_SCORE_MIN = 300
TARGET_SCORE_MAX = 10000

PLAY_REMAINING_MAX = 5
DISCARD_REMAINING_MAX = 4

PLAYER_HAND_SIZE_MIN = 5
PLAYER_HAND_SIZE_MAX = 12

HAND_LEVEL_MAX = 20  # upper bound per-hand level in :func:`generate_snapshot`


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
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _card_to_dict(card: Card) -> dict:
    return {"card_id": card.card_id, "enhancement": card.enhancement, "edition": card.edition}


def _dict_to_card(d: dict) -> Card:
    return Card(card_id=d["card_id"], enhancement=d["enhancement"], edition=d["edition"])


def _joker_to_dict(joker: Joker) -> dict:
    return {"id": joker.id, "edition": joker.edition}


def _dict_to_joker(d: dict) -> Joker:
    return Joker(id=d["id"], edition=d["edition"])


def hand_levels_from_levels(levels: dict[int, int]) -> dict[int, int]:
    """Normalize a ``hand_type_id -> level`` map (copy with int keys)."""
    return {int(k): int(v) for k, v in levels.items()}


def hand_level_from_chips_mult(hand: HandType, chips: int, mult: float) -> int:
    """Invert :func:`util.chips_mult_for_hand_level` for wiki grid pairs; used for legacy JSON."""
    bc, bm = POKER_HAND_BASE_CHIPS_MULT[hand]
    dc, dm = POKER_HAND_LEVEL_UP_CHIPS_MULT[hand]
    mf = float(mult)
    if chips < bc or mf < float(bm):
        raise ValueError(
            f"{hand!s}: chips/mult ({chips}, {mult}) below base ({bc}, {bm})"
        )
    rem_c = chips - bc
    rem_m = mf - float(bm)
    if dc == 0:
        if rem_c != 0:
            raise ValueError(f"{hand!s}: chips offset {rem_c} but Δchips is 0")
        l_c = 1
    else:
        if rem_c % dc != 0:
            raise ValueError(
                f"{hand!s}: chips {chips} is not base + n*{dc} for integer n (base {bc})"
            )
        l_c = 1 + rem_c // dc
    if dm == 0:
        if abs(rem_m) > 1e-9:
            raise ValueError(f"{hand!s}: mult offset {rem_m} but Δmult is 0")
        l_m = 1
    else:
        q = rem_m / float(dm)
        if abs(q - round(q)) > 1e-9:
            raise ValueError(
                f"{hand!s}: mult {mult} is not base + n*{dm} for integer n (base {bm})"
            )
        l_m = 1 + int(round(q))
    if l_c != l_m:
        raise ValueError(
            f"{hand!s}: chips implies level {l_c}, mult implies level {l_m} "
            f"for (chips, mult)=({chips}, {mult})"
        )
    return l_c


def _legacy_hand_levels_row_to_level(hand_type_id: int, chips: int, hm_obs: int) -> int:
    """Legacy ``[chips, hm_obs]`` row → level (``hm_obs`` is scoring/obs additive mult column)."""
    mw = 1 if hm_obs == 0 else hm_obs
    return hand_level_from_chips_mult(HandType(hand_type_id), chips, float(mw))


def _hand_levels_dict_to_internal(raw: dict) -> dict[int, int]:
    """JSON ``hand_levels``: per-key either a level (int) or legacy ``[chips, mult_obs]``."""
    out: dict[int, int] = {}
    for k, v in raw.items():
        ki = int(k)
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(
                    f"hand_levels[{k!r}] legacy form must be [chips, mult], got {v!r}"
                )
            out[ki] = _legacy_hand_levels_row_to_level(
                ki, int(v[0]), int(float(v[1]))
            )
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            lev = int(v)
            if lev < 1:
                raise ValueError(f"hand_levels[{k!r}] level must be >= 1, got {lev}")
            out[ki] = lev
        else:
            raise TypeError(
                f"hand_levels[{k!r}] must be a level (number) or [chips, mult], got {type(v).__name__}"
            )
    return out


def _hand_levels_snapshot_to_json(snapshot: GameSnapshot) -> dict[str, int]:
    """Serialize ``GameSnapshot.hand_levels`` (levels) to JSON string keys."""
    return {str(k): int(v) for k, v in snapshot.hand_levels.items()}


def snapshot_to_dict(snapshot: GameSnapshot) -> dict:
    """Convert a ``GameSnapshot`` to a JSON-serializable dict."""
    return {
        "target_score": snapshot.target_score,
        "current_score": snapshot.current_score,
        "blind_id": snapshot.blind_id,
        "hand": [_card_to_dict(c) for c in snapshot.hand],
        "deck": [_card_to_dict(c) for c in snapshot.deck],
        "jokers": [_joker_to_dict(j) for j in snapshot.jokers],
        "play_remaining": snapshot.play_remaining,
        "discard_remaining": snapshot.discard_remaining,
        "player_hand_size": snapshot.player_hand_size,
        "hand_levels": _hand_levels_snapshot_to_json(snapshot),
    }


def dict_to_snapshot(d: dict) -> GameSnapshot:
    """Reconstruct a ``GameSnapshot`` from a dict (e.g. parsed from JSON)."""
    hl_raw = d.get("hand_levels", {})
    if not isinstance(hl_raw, dict):
        raise TypeError(f"hand_levels must be a dict, got {type(hl_raw).__name__}")
    return GameSnapshot(
        target_score=d["target_score"],
        current_score=d["current_score"],
        blind_id=d["blind_id"],
        hand=[_dict_to_card(c) for c in d["hand"]],
        deck=[_dict_to_card(c) for c in d["deck"]],
        jokers=[_dict_to_joker(j) for j in d["jokers"]],
        play_remaining=d["play_remaining"],
        discard_remaining=d["discard_remaining"],
        player_hand_size=d["player_hand_size"],
        hand_levels=_hand_levels_dict_to_internal(hl_raw),
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_snapshot(path: Union[str, Path], snapshot: GameSnapshot) -> None:
    """Write one ``GameSnapshot`` as a single JSON object (one snapshot per file)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot_to_dict(snapshot), f, indent=2, ensure_ascii=False)


def load_snapshot(path: Union[str, Path]) -> GameSnapshot:
    """Load a ``GameSnapshot`` from a JSON file whose root is one object (not an array)."""
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
    """Load every file matching ``pattern`` under ``directory`` via :func:`load_snapshot`."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    paths = sorted(d.glob(pattern)) if sort else list(d.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {d}")
    return [load_snapshot(p) for p in paths]


def save_snapshots(snapshots: Union[GameSnapshot, List[GameSnapshot]], path: Union[str, Path]) -> None:
    """Save snapshots to a JSON file as a **JSON array** (legacy batch format).

    For training pools (one snapshot per ``*.json``), use :func:`save_snapshot` per file and
    :func:`load_snapshot` / :func:`load_snapshot_pool_from_json_dir`.
    """
    if isinstance(snapshots, GameSnapshot):
        snapshots = [snapshots]
    data = [snapshot_to_dict(s) for s in snapshots]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    snapshots = generate_snapshots(3)
    out_dir = Path("snapshots_out")
    out_dir.mkdir(exist_ok=True)
    for i, s in enumerate(snapshots):
        save_snapshot(out_dir / f"{i}.json", s)
    print(f"Saved {len(snapshots)} snapshots under {out_dir}/")

    loaded0 = load_snapshot(out_dir / "0.json")
    print(f"Loaded snapshot 0: target_score={loaded0.target_score}")
