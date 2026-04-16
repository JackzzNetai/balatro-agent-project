"""Stress / edge-case tests for ``recognize_poker_hand``."""

import random

import pytest

from defs import Edition, HandType
from engine import Card
from util import card_id_from_suit_rank, recognize_poker_hand


def card(suit: int, rank: int, enhancement: int = 0, edition: int = 0) -> Card:
    return Card(card_id_from_suit_rank(suit, rank), enhancement, edition)


def classify(cards: list[Card]) -> HandType:
    return recognize_poker_hand(cards)[0]


# --- every vanilla 5-card straight (mixed suits -> Straight not SF) ---


@pytest.mark.parametrize(
    "ranks",
    [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
        [0, 9, 10, 11, 12],
    ],
)
def test_straight_mixed_suits_for_each_valid_rank_set(ranks):
    suits = [0, 1, 2, 3, 0]
    played = [card(suits[i], ranks[i]) for i in range(5)]
    assert classify(played) == HandType.STRAIGHT


@pytest.mark.parametrize(
    "ranks",
    [
        [1, 2, 3, 4, 6],
        [0, 2, 3, 4, 5],
        [8, 9, 10, 11, 0],
        [0, 1, 2, 3, 5],
    ],
)
def test_not_straight_gap_or_bad_ace_wrap(ranks):
    suits = [0, 1, 2, 3, 0]
    played = [card(suits[i], ranks[i]) for i in range(5)]
    assert classify(played) == HandType.HIGH_CARD


# --- every 5-card straight flush (all clubs) including wheel & broadway ---


@pytest.mark.parametrize(
    "ranks",
    [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
        [0, 9, 10, 11, 12],
    ],
)
def test_straight_flush_all_clubs(ranks):
    played = [card(0, r) for r in ranks]
    assert classify(played) == HandType.STRAIGHT_FLUSH


# --- flush without straight: many gaps ---


def test_flush_with_largest_possible_gaps_still_not_straight():
    played = [card(2, r) for r in (0, 2, 4, 6, 10)]
    assert classify(played) == HandType.FLUSH


# --- n=1..4 marginal types ---


@pytest.mark.parametrize(
    "played,expected",
    [
        ([card(0, 5)], HandType.HIGH_CARD),
        ([card(0, 5), card(1, 7)], HandType.HIGH_CARD),
        ([card(0, 5), card(1, 5)], HandType.PAIR),
        ([card(0, 3), card(1, 3), card(2, 3)], HandType.THREE_OF_A_KIND),
        ([card(0, 4), card(1, 4), card(2, 4), card(3, 4)], HandType.FOUR_OF_A_KIND),
    ],
)
def test_short_hands_marginal(played, expected):
    assert classify(played) == expected


def test_n4_trips_plus_kicker():
    played = [card(0, 6), card(1, 6), card(2, 6), card(3, 2)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.THREE_OF_A_KIND
    assert len(scored) == 3
    assert scored == played[:3]


def test_n4_one_pair_three_kickers():
    played = [card(0, 8), card(1, 8), card(2, 1), card(3, 2)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.PAIR
    assert len(scored) == 2
    assert scored == played[:2]


def test_n5_trips_two_kickers_not_full_house():
    played = [card(0, 0), card(1, 0), card(2, 0), card(3, 11), card(0, 12)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.THREE_OF_A_KIND
    assert len(scored) == 3
    assert scored == played[:3]


def test_n5_two_pair_aces_and_kings_queen_kicker():
    played = [card(0, 0), card(1, 0), card(2, 12), card(3, 12), card(0, 11)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.TWO_PAIR
    assert len(scored) == 4
    assert scored == played[:4]


# --- compound precedence ---


def test_flush_beats_straight_when_both_possible_separate_hands_not_applicable():
    """Single multiset: cannot be both without being SF; sanity that flush-only exists."""
    played = [card(0, r) for r in (0, 2, 4, 6, 8)]
    assert classify(played) == HandType.FLUSH


def test_straight_flush_beats_flush_same_components_impossible():
    """SF when straight+flush; one hand cannot be ambiguous."""
    played = [card(0, i) for i in range(5)]
    assert classify(played) == HandType.STRAIGHT_FLUSH


# --- enhancements / editions ignored except Wild ---


@pytest.mark.parametrize("enh", [0, 1, 2, 4, 5])
def test_pair_invariant_under_non_wild_enhancements(enh):
    played = [
        Card(card_id_from_suit_rank(0, 7), enh, 0),
        Card(card_id_from_suit_rank(1, 7), enh, 0),
    ]
    assert classify(played) == HandType.PAIR


@pytest.mark.parametrize("ed", list(Edition))
def test_high_card_invariant_under_edition(ed):
    played = [Card(card_id_from_suit_rank(0, 4), 0, int(ed))]
    assert classify(played) == HandType.HIGH_CARD


# --- Wild edge cases ---


def test_all_five_wild_same_rank_duplicates_flush_five():
    played = [Card(0, 3, 0) for _ in range(5)]
    assert classify(played) == HandType.FLUSH_FIVE


def test_two_wilds_bridge_flush_three_natural_clubs_not_straight():
    """Flush only: ranks have a gap so not a straight / straight flush."""
    played = [
        card(0, 5),
        card(0, 6),
        card(0, 7),
        card(1, 9, enhancement=3),
        card(2, 10, enhancement=3),
    ]
    assert classify(played) == HandType.FLUSH


def test_two_wilds_straight_flush_broadway():
    played = [
        card(0, 9),
        card(0, 10),
        card(0, 11),
        card(0, 12),
        card(1, 0, enhancement=3),
    ]
    assert classify(played) == HandType.STRAIGHT_FLUSH


def test_wild_cannot_fix_conflicting_two_natural_suit_anchors():
    """Two non-Wild groups in different suits -> no flush even with one Wild."""
    played = [
        card(0, 2),
        card(0, 3),
        card(1, 4),
        card(1, 5),
        card(2, 6, enhancement=3),
    ]
    ranks = [2, 3, 4, 5, 6]
    assert ranks == sorted(ranks) and ranks[-1] - ranks[0] == 4
    assert classify(played) == HandType.STRAIGHT


def test_flush_house_with_one_wild():
    played = [
        card(0, 2),
        card(0, 2),
        card(0, 2),
        card(0, 11),
        card(1, 11, enhancement=3),
    ]
    assert classify(played) == HandType.FLUSH_HOUSE


# --- duplicate card_id multiset ---


def test_duplicate_ids_five_of_a_kind():
    played = [Card(0, 0, 0), Card(13, 0, 0), Card(26, 0, 0), Card(39, 0, 0), Card(0, 0, 0)]
    assert classify(played) == HandType.FIVE_OF_A_KIND


def test_duplicate_ids_flush_five():
    played = [Card(7, 0, 0) for _ in range(5)]
    assert classify(played) == HandType.FLUSH_FIVE


# --- order independence (wiki: order does not affect type) ---


def test_order_independence_random_permutations_full_house():
    base = [
        card(0, 3),
        card(1, 3),
        card(2, 3),
        card(0, 8),
        card(1, 8),
    ]
    rng = random.Random(0)
    for _ in range(30):
        perm = base[:]
        rng.shuffle(perm)
        assert classify(perm) == HandType.FULL_HOUSE


def test_order_independence_two_pair():
    base = [card(s, r) for s, r in ((0, 4), (1, 4), (0, 9), (2, 9), (3, 1))]
    rng = random.Random(1)
    for _ in range(30):
        perm = base[:]
        rng.shuffle(perm)
        assert classify(perm) == HandType.TWO_PAIR


# --- four of a kind does not classify as two pair ---


def test_quads_plus_kicker_not_two_pair():
    played = [card(0, 10), card(1, 10), card(2, 10), card(3, 10), card(0, 2)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FOUR_OF_A_KIND
    assert len(scored) == 4
    assert scored == played[:4]


# --- full house beats trips pattern ---


def test_full_house_not_three_of_a_kind():
    played = [card(0, 5), card(1, 5), card(2, 5), card(0, 9), card(1, 9)]
    assert classify(played) == HandType.FULL_HOUSE


# --- high card n=5 all distinct non-straight ---


def test_five_distinct_non_straight_high_card():
    played = [card(i % 4, r) for i, r in enumerate([0, 2, 5, 7, 10])]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert len(scored) == 1
    assert scored[0] is played[0]


# --- straight that would be broadway if ace counted wrong ---


def test_mixed_suit_almost_broadway_missing_one_rank_is_high_card():
    played = [
        card(0, 0),
        card(1, 9),
        card(2, 10),
        card(3, 11),
        card(0, 8),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert len(scored) == 1
    assert scored[0] is played[0]
