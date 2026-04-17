"""Balatro wiki poker hand base + level-up tables."""

from __future__ import annotations

import pytest

from defs import HandType, POKER_HAND_BASE_CHIPS_MULT, POKER_HAND_LEVEL_UP_CHIPS_MULT
from env.snapshot_io import hand_level_from_chips_mult, hand_levels_from_levels
from util import chips_mult_for_hand_level


@pytest.mark.parametrize(
    "hand,base_c,base_m",
    [
        (HandType.HIGH_CARD, 5, 1),
        (HandType.PAIR, 10, 2),
        (HandType.THREE_OF_A_KIND, 30, 3),
        (HandType.STRAIGHT_FLUSH, 100, 8),
        (HandType.FLUSH_FIVE, 160, 16),
    ],
)
def test_level_1_matches_wiki_base(hand: HandType, base_c: int, base_m: int):
    c, m = chips_mult_for_hand_level(hand, 1)
    assert (c, m) == (base_c, base_m)
    assert POKER_HAND_BASE_CHIPS_MULT[hand] == (base_c, base_m)


def test_level_2_one_upgrade_high_card():
    c, m = chips_mult_for_hand_level(HandType.HIGH_CARD, 2)
    assert (c, m) == (15, 2)
    dc, dm = POKER_HAND_LEVEL_UP_CHIPS_MULT[HandType.HIGH_CARD]
    assert (dc, dm) == (10, 1)


def test_level_2_three_of_a_kind():
    c, m = chips_mult_for_hand_level(HandType.THREE_OF_A_KIND, 2)
    assert (c, m) == (50, 5)


def test_level_2_straight_flush():
    c, m = chips_mult_for_hand_level(HandType.STRAIGHT_FLUSH, 2)
    assert (c, m) == (140, 12)


def test_hand_levels_from_levels_round_trip():
    levels = {int(h): 3 for h in HandType}
    hl = hand_levels_from_levels(levels)
    assert hl == levels
    for k, lev in hl.items():
        c, mw = chips_mult_for_hand_level(HandType(k), lev)
        assert hand_level_from_chips_mult(HandType(k), c, float(mw)) == lev


def test_hand_level_from_chips_mult_rejects_off_grid():
    with pytest.raises(ValueError, match="not base"):
        hand_level_from_chips_mult(HandType.HIGH_CARD, 10, 1)


def test_chips_mult_rejects_level_zero():
    with pytest.raises(ValueError, match=">= 1"):
        chips_mult_for_hand_level(HandType.PAIR, 0)
