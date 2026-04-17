"""Balatro poker hand type ids (for ``hand_levels`` rows and recognition).

Ids follow the usual in-game progression from weakest to strongest **among the
twelve tracked hands**. **Royal Flush** uses the same id as **Straight Flush**
(they share base scoring and level-ups per
https://balatrowiki.org/w/Poker_Hands — Neptune planet, equivalent for levels).

Regular hands occupy ids ``0``–``7``; **Straight Flush** is ``8`` (includes royal).
Secret hands (**Five of a Kind**, **Flush House**, **Flush Five**) are ``9``–``11``.

Prefer ``from defs import ...`` for the stable public API.
"""

from __future__ import annotations

from enum import IntEnum

# Keep in sync with observation ``hand_levels`` layout (see ``environment``).
HAND_TYPE_COUNT = 12


class HandType(IntEnum):
    """Integer hand type; order matches Balatro wiki *Regular* then *Secret* rows.

    Royal flush is **not** a separate member: classify and score as
    :attr:`STRAIGHT_FLUSH`.
    """

    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_OF_A_KIND = 9
    FLUSH_HOUSE = 10
    FLUSH_FIVE = 11


# Base chips and mult at level 1 (Balatro wiki “Base Scoring”).
POKER_HAND_BASE_CHIPS_MULT: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}

# Per level above 1: add (Δchips, Δmult) from each hand’s planet (wiki planet rows).
POKER_HAND_LEVEL_UP_CHIPS_MULT: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_OF_A_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.FIVE_OF_A_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}


# Human-readable labels (wiki English names).
HAND_TYPE_LABELS: dict[HandType, str] = {
    HandType.HIGH_CARD: "High Card",
    HandType.PAIR: "Pair",
    HandType.TWO_PAIR: "Two Pair",
    HandType.THREE_OF_A_KIND: "Three of a Kind",
    HandType.STRAIGHT: "Straight",
    HandType.FLUSH: "Flush",
    HandType.FULL_HOUSE: "Full House",
    HandType.FOUR_OF_A_KIND: "Four of a Kind",
    HandType.STRAIGHT_FLUSH: "Straight Flush",
    HandType.FIVE_OF_A_KIND: "Five of a Kind",
    HandType.FLUSH_HOUSE: "Flush House",
    HandType.FLUSH_FIVE: "Flush Five",
}
