"""Optional debug / REPL helpers for ``balatro_lite_gym``. Not used in training hot paths."""

from __future__ import annotations

from typing import Literal, TextIO

from defs import (
    BOSS_BLIND_LABELS,
    CARD_ENHANCEMENT_LABELS,
    CARD_RANK_LABELS,
    EDITION_LABELS,
    HAND_TYPE_LABELS,
    JOKER_LABELS,
    NO_BOSS_BLIND_ID,
    BossBlind,
    CardEnhancement,
    Edition,
    HandType,
    JokerId,
)
from engine import Card, GameSnapshot, Joker
from util import rank_from_card_id, suit_from_card_id

_SUIT_GLYPH: dict[int, str] = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
_DECK_PREVIEW = 8

__all__ = ["format_snapshot", "print_snapshot"]


def _blind_label(blind_id: int) -> str:
    if blind_id == NO_BOSS_BLIND_ID:
        return "non-boss blind (Small/Big)"
    return BOSS_BLIND_LABELS[BossBlind(blind_id)]


def _format_card_line(idx: int, c: Card) -> str:
    rank = rank_from_card_id(c.card_id)
    suit_i = int(suit_from_card_id(c.card_id))
    glyph = _SUIT_GLYPH[suit_i]
    face = CARD_RANK_LABELS[rank]
    parts = [f"  [{idx}] {face}{glyph}  id={c.card_id}"]
    enh = CardEnhancement(c.enhancement)
    if enh != CardEnhancement.NONE:
        parts.append(f" enh={CARD_ENHANCEMENT_LABELS[enh]}")
    ed = Edition(c.edition)
    if ed != Edition.BASE:
        parts.append(f" ed={EDITION_LABELS[ed]}")
    return "".join(parts)


def _format_joker_line(idx: int, j: Joker) -> str:
    try:
        name = JOKER_LABELS[JokerId(j.id)]
    except ValueError:
        name = f"(unknown id {j.id})"
    parts = [f"  [{idx}] {name}  id={j.id}"]
    ed = Edition(j.edition)
    if ed != Edition.BASE:
        parts.append(f" ed={EDITION_LABELS[ed]}")
    return "".join(parts)


def format_snapshot(
    snapshot: GameSnapshot,
    *,
    deck: Literal["summary", "full"] = "summary",
) -> str:
    """Return a multi-line human-readable dump of ``snapshot`` (``GameSnapshot`` fields only)."""
    lines: list[str] = [
        "=== GameSnapshot ===",
        f"target_score:      {snapshot.target_score}",
        f"current_score:     {snapshot.current_score}",
        f"blind_id:          {snapshot.blind_id}  ({_blind_label(snapshot.blind_id)})",
        f"play_remaining:    {snapshot.play_remaining}",
        f"discard_remaining: {snapshot.discard_remaining}",
        f"player_hand_size:  {snapshot.player_hand_size}",
        "",
        "--- Hand ---",
    ]
    if not snapshot.hand:
        lines.append("  (empty)")
    else:
        for i, c in enumerate(snapshot.hand):
            lines.append(_format_card_line(i, c))

    lines.extend(["", "--- Deck ---"])
    n_deck = len(snapshot.deck)
    if n_deck == 0:
        lines.append("  (empty)")
    elif deck == "full":
        for i, c in enumerate(snapshot.deck):
            lines.append(_format_card_line(i, c))
    else:
        lines.append(f"  {n_deck} card(s)")
        preview = min(_DECK_PREVIEW, n_deck)
        if preview:
            lines.append(f"  (first {preview} shown)")
            for i, c in enumerate(snapshot.deck[:preview]):
                lines.append(_format_card_line(i, c))

    lines.extend(["", "--- Jokers ---"])
    if not snapshot.jokers:
        lines.append("  (none)")
    else:
        for i, j in enumerate(snapshot.jokers):
            lines.append(_format_joker_line(i, j))

    lines.extend(["", "--- hand_levels ---"])
    if not snapshot.hand_levels:
        lines.append("  (none)")
    else:
        for ht in sorted(snapshot.hand_levels.keys()):
            chips, mult = snapshot.hand_levels[ht]
            label = HAND_TYPE_LABELS[HandType(ht)]
            lines.append(f"  {label}  chips={chips}  mult={mult}")

    return "\n".join(lines) + "\n"


def print_snapshot(
    snapshot: GameSnapshot,
    *,
    file: TextIO | None = None,
    deck: Literal["summary", "full"] = "summary",
) -> None:
    """Print :func:`format_snapshot` to ``file`` (default stdout)."""
    print(format_snapshot(snapshot, deck=deck), end="", file=file)
