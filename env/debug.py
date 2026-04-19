"""Optional debug / REPL helpers for snapshots and combat state. Not used in training hot paths."""

from __future__ import annotations

import sys
from typing import Literal, TextIO

from defs import (
    BOSS_BLIND_LABELS,
    CARD_ENHANCEMENT_LABELS,
    CARD_RANK_LABELS,
    EDITION_LABELS,
    HAND_TYPE_COUNT,
    HAND_TYPE_LABELS,
    NO_BOSS_BLIND_ID,
    BossBlind,
    CardEnhancement,
    CardRank,
    CardSuit,
    Edition,
    HandType,
    JOKER_LABELS,
    JokerId,
    NUM_RANKS,
    NUM_SUITS,
)
from engine import Card, GameSnapshot, Joker
from util import (
    chips_mult_for_hand_level,
    hand_debuff_mask,
    rank_from_card_id,
    suit_from_card_id,
)

_SUIT_GLYPH: dict[int, str] = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}

__all__ = [
    "format_snapshot",
    "print_snapshot",
]


def _blind_label(blind_id: int) -> str:
    if blind_id == NO_BOSS_BLIND_ID:
        return "non-boss blind (Small/Big)"
    return BOSS_BLIND_LABELS[BossBlind(blind_id)]


def _card_face(card_id: int) -> str:
    rank = rank_from_card_id(card_id)
    suit_i = int(suit_from_card_id(card_id))
    return f"{CARD_RANK_LABELS[rank]}{_SUIT_GLYPH[suit_i]}"


def _enhancement_str(c: Card) -> str:
    enh = CardEnhancement(c.enhancement)
    return "" if enh is CardEnhancement.NONE else CARD_ENHANCEMENT_LABELS[enh]


def _edition_str(c: Card) -> str:
    ed = Edition(c.edition)
    return "" if ed is Edition.BASE else EDITION_LABELS[ed]


def _cell(s: str, w: int) -> str:
    if len(s) <= w:
        return s.center(w)
    return s[: max(1, w - 1)] + "…"


def _format_hand_table(hand: list[Card], debuff_mask: list[bool]) -> list[str]:
    """Column-per-card layout (same idea as ``print_combat_state`` hand block)."""
    lines: list[str] = []
    if not hand:
        lines.append("  (no cards in hand)")
        lines.append("  debuff (util.hand_debuff_mask): (no slots)")
        return lines

    if len(debuff_mask) != len(hand):
        raise ValueError(
            f"hand_debuff_mask length {len(debuff_mask)} != len(hand) {len(hand)}"
        )

    cols: list[dict[str, str]] = []
    for i, c in enumerate(hand):
        debuffed = bool(debuff_mask[i])
        cols.append(
            {
                "slot": str(i),
                "card": _card_face(c.card_id),
                "enhancement": _enhancement_str(c),
                "edition": _edition_str(c),
                "debuff": "❌" if debuffed else "",
            },
        )

    row_spec: list[tuple[str, str]] = [
        ("slot", "slot"),
        ("card", "card"),
        ("enhancement", "enhancement"),
        ("edition", "edition"),
        ("debuff", "debuff"),
    ]
    w_label = max(len(lbl) for _, lbl in row_spec)
    n = len(cols)
    col_widths: list[int] = []
    for j in range(n):
        w = max(len(cols[j]["card"]), *(len(cols[j][key]) for key, _ in row_spec))
        col_widths.append(max(2, w))

    gap = " │ "

    def _hline(left_w: int) -> str:
        s = "  " + "─" * left_w + "─┼"
        for j, cw in enumerate(col_widths):
            s += "─" * (cw + 2)
            if j < n - 1:
                s += "┼"
        return s

    hdr_cells = [_cell(cols[j]["card"], col_widths[j]) for j in range(n)]
    lines.append(f"  {' ':{w_label}}{gap}{gap.join(hdr_cells)}")
    lines.append(_hline(w_label))
    for key, lbl in row_spec:
        cells = [_cell(cols[j][key], col_widths[j]) for j in range(n)]
        lines.append(f"  {lbl:{w_label}}{gap}{gap.join(cells)}")
    return lines


def _format_deck_grid(deck: list[Card]) -> list[str]:
    """4×13 suit × rank counts (``card_id = suit * 13 + rank``)."""
    lines: list[str] = []
    n_deck = len(deck)
    lines.append(f"  cards in draw pile: {n_deck}")

    grid: list[list[int]] = [[0] * NUM_RANKS for _ in range(NUM_SUITS)]
    for c in deck:
        cid = int(c.card_id)
        if 0 <= cid <= 51:
            grid[cid // NUM_RANKS][cid % NUM_RANKS] += 1

    inner = 3
    ncols = 1 + NUM_RANKS
    indent = "  "

    def _deck_slot(text: str) -> str:
        t = (text or "")[:inner]
        return f"{t:^{inner}}"

    def _deck_rule(left: str, cross: str, right: str) -> str:
        return indent + left + cross.join("─" * inner for _ in range(ncols)) + right

    lines.append(_deck_rule("┌", "┬", "┐"))
    hdr_cells = [""] + [CARD_RANK_LABELS[CardRank(r)] for r in range(NUM_RANKS)]
    lines.append(indent + "│" + "│".join(_deck_slot(x) for x in hdr_cells) + "│")
    lines.append(_deck_rule("├", "┼", "┤"))
    for s in range(NUM_SUITS):
        sym = _SUIT_GLYPH[s]
        row_cells = [_deck_slot(sym)] + [
            _deck_slot("" if grid[s][ri] == 0 else str(grid[s][ri])) for ri in range(NUM_RANKS)
        ]
        lines.append(indent + "│" + "│".join(row_cells) + "│")
    lines.append(_deck_rule("└", "┴", "┘"))
    return lines


def _format_joker_slot(idx: int, j: Joker) -> str:
    try:
        name = JOKER_LABELS[JokerId(j.id)]
    except ValueError:
        name = f"(unknown id {j.id})"
    line = f"  j{idx}: {name} (id={j.id})"
    ed = Edition(j.edition)
    if ed is not Edition.BASE:
        line += f" · {EDITION_LABELS[ed]}"
    return line


def format_snapshot(
    snapshot: GameSnapshot,
    *,
    deck: Literal["summary", "full"] = "summary",
) -> str:
    """Return a human-readable dump of ``GameSnapshot`` (layout similar to ``print_combat_state``)."""
    lines: list[str] = []

    cur, tgt = snapshot.current_score, snapshot.target_score
    need = max(0, int(tgt) - int(cur))
    lines.append("")
    lines.append("=== State ===")
    lines.append(f"  round chips: {cur} / target {tgt}  (need {need} more)")
    lines.append(
        f"  hand_size={snapshot.player_hand_size}  hands_left={snapshot.play_remaining}  "
        f"discards_left={snapshot.discard_remaining}  blind_id={snapshot.blind_id}  "
        f"({_blind_label(snapshot.blind_id)})"
    )

    lines.append("")
    lines.append("=== Hand ===")
    _debuff = hand_debuff_mask(snapshot)
    lines.extend(_format_hand_table(snapshot.hand, _debuff))

    lines.append("")
    lines.append("=== Jokers ===")
    if not snapshot.jokers:
        lines.append("  (none)")
    else:
        for i, j in enumerate(snapshot.jokers):
            lines.append(_format_joker_slot(i, j))

    lines.append("")
    lines.append("=== Draw pile ===")
    lines.extend(_format_deck_grid(snapshot.deck))
    if deck == "full" and snapshot.deck:
        lines.append("  --- full list ---")
        for i, c in enumerate(snapshot.deck):
            rank = rank_from_card_id(c.card_id)
            suit_i = int(suit_from_card_id(c.card_id))
            glyph = _SUIT_GLYPH[suit_i]
            face = CARD_RANK_LABELS[rank]
            extra = []
            enh = CardEnhancement(c.enhancement)
            if enh != CardEnhancement.NONE:
                extra.append(f"enh={CARD_ENHANCEMENT_LABELS[enh]}")
            ed = Edition(c.edition)
            if ed != Edition.BASE:
                extra.append(f"ed={EDITION_LABELS[ed]}")
            suf = ("  " + " ".join(extra)) if extra else ""
            lines.append(f"  [{i}] {face}{glyph}  id={c.card_id}{suf}")

    lines.append("")
    lines.append("=== Hand levels ===")
    for hi in range(HAND_TYPE_COUNT):
        name = HAND_TYPE_LABELS[HandType(hi)]
        lev = snapshot.hand_levels.get(hi)
        if lev is None:
            lines.append(f"  {name:16s}  #{hi}  lvl=—  chips=—  mult=—")
        else:
            ch, mu = chips_mult_for_hand_level(HandType(hi), lev)
            lines.append(
                f"  {name:16s}  #{hi}  lvl={lev:3d}  chips={ch:5d}  mult={mu:3d}"
            )

    lines.append("")
    return "\n".join(lines)


def print_snapshot(
    snapshot: GameSnapshot,
    *,
    file: TextIO | None = None,
    deck: Literal["summary", "full"] = "summary",
) -> None:
    """Print :func:`format_snapshot` to ``file`` (default ``sys.stdout``)."""
    if file is None:
        file = sys.stdout
    print(format_snapshot(snapshot, deck=deck), end="", file=file)
