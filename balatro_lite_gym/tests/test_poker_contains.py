"""Stress and near-miss tests for ``util.played_contains`` (wiki 'contains' semantics)."""

import pytest

from defs import CardEnhancement, HandType
from engine import Card
from util import card_id_from_suit_rank, played_contains, recognize_poker_hand


def C(card_id: int, enhancement: int = 0) -> Card:
    return Card(card_id, enhancement, 0)


def _best(played: list[Card]) -> HandType:
    return recognize_poker_hand(played)[0]


# --- rank / contains vs best hand ---


def test_trips_contains_pair_and_trips_not_two_pair():
    played = [C(0), C(13), C(26)]  # three aces
    assert _best(played) is HandType.THREE_OF_A_KIND
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.THREE_OF_A_KIND) is True
    assert played_contains(played, HandType.TWO_PAIR) is False
    assert played_contains(played, HandType.FOUR_OF_A_KIND) is False


def test_full_house_contains_pair_and_trips_not_two_pair():
    played = [C(0), C(13), C(26), C(1), C(14)]  # AAA + pair of twos
    assert _best(played) is HandType.FULL_HOUSE
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.THREE_OF_A_KIND) is True
    assert played_contains(played, HandType.FULL_HOUSE) is True
    assert played_contains(played, HandType.TWO_PAIR) is False


def test_two_pair_four_cards_contains_pair_and_two_pair():
    played = [C(0), C(13), C(1), C(14)]
    assert _best(played) is HandType.TWO_PAIR
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.TWO_PAIR) is True
    assert played_contains(played, HandType.THREE_OF_A_KIND) is False


def test_single_ace_high_card_contains_high_card_only():
    played = [C(0)]
    assert _best(played) is HandType.HIGH_CARD
    assert played_contains(played, HandType.HIGH_CARD) is True
    assert played_contains(played, HandType.PAIR) is False


def test_pair_hand_does_not_contain_high_card_pattern():
    played = [C(0), C(13)]
    assert _best(played) is HandType.PAIR
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.HIGH_CARD) is False


def test_quads_contains_weaker_rank_components_not_two_pair():
    played = [C(0), C(13), C(26), C(39), C(1)]
    assert _best(played) is HandType.FOUR_OF_A_KIND
    assert played_contains(played, HandType.FOUR_OF_A_KIND) is True
    assert played_contains(played, HandType.THREE_OF_A_KIND) is True
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.TWO_PAIR) is False


# --- straight / flush near misses ---


def test_gap_ranks_not_straight_even_if_almost_wheel():
    """A,2,3,4,6 — gap at 5; mixed suits so not a flush."""
    played = [
        C(card_id_from_suit_rank(0, 0)),
        C(card_id_from_suit_rank(1, 1)),
        C(card_id_from_suit_rank(2, 2)),
        C(card_id_from_suit_rank(3, 3)),
        C(card_id_from_suit_rank(0, 5)),
    ]
    assert _best(played) is HandType.HIGH_CARD
    assert played_contains(played, HandType.STRAIGHT) is False
    assert played_contains(played, HandType.FLUSH) is False


def test_wheel_straight_mixed_suits():
    played = [
        C(card_id_from_suit_rank(0, 0)),
        C(card_id_from_suit_rank(1, 1)),
        C(card_id_from_suit_rank(2, 2)),
        C(card_id_from_suit_rank(3, 3)),
        C(card_id_from_suit_rank(0, 4)),
    ]
    assert _best(played) is HandType.STRAIGHT
    assert played_contains(played, HandType.STRAIGHT) is True
    assert played_contains(played, HandType.FLUSH) is False


def test_broadway_straight_mixed_suits():
    ranks = [8, 9, 10, 11, 12]
    played = [C(card_id_from_suit_rank(i % 4, r)) for i, r in enumerate(ranks)]
    assert _best(played) is HandType.STRAIGHT
    assert played_contains(played, HandType.STRAIGHT) is True


def test_four_natural_flush_miss_with_one_off_suit():
    """Four spades + one heart, no Wild — no flush."""
    played = [
        C(card_id_from_suit_rank(3, 0)),
        C(card_id_from_suit_rank(3, 2)),
        C(card_id_from_suit_rank(3, 4)),
        C(card_id_from_suit_rank(3, 6)),
        C(card_id_from_suit_rank(2, 8)),
    ]
    assert played_contains(played, HandType.FLUSH) is False
    assert played_contains(played, HandType.STRAIGHT_FLUSH) is False


def test_straight_flush_implies_straight_and_flush_contains():
    W = int(CardEnhancement.WILD)
    played = [
        Card(2, 0, 0),
        Card(3, 0, 0),
        Card(4, 0, 0),
        Card(5, 0, 0),
        Card(19, W, 0),
    ]
    assert _best(played) is HandType.STRAIGHT_FLUSH
    assert played_contains(played, HandType.STRAIGHT_FLUSH) is True
    assert played_contains(played, HandType.STRAIGHT) is True
    assert played_contains(played, HandType.FLUSH) is True


def test_conflicting_natural_suits_straight_not_flush():
    """Mirror stress: straight possible, two suit anchors -> no flush."""
    played = [
        C(card_id_from_suit_rank(0, 2)),
        C(card_id_from_suit_rank(0, 3)),
        C(card_id_from_suit_rank(1, 4)),
        C(card_id_from_suit_rank(1, 5)),
        Card(card_id_from_suit_rank(2, 6), int(CardEnhancement.WILD), 0),
    ]
    assert _best(played) is HandType.STRAIGHT
    assert played_contains(played, HandType.STRAIGHT) is True
    assert played_contains(played, HandType.FLUSH) is False


# --- short hands ---


def test_three_cards_straight_and_flush_contains_false():
    played = [C(0), C(13), C(26)]
    assert played_contains(played, HandType.STRAIGHT) is False
    assert played_contains(played, HandType.FLUSH) is False


# --- secret hands vs weaker inner ---


def test_flush_five_contains_flush_five_not_five_of_a_kind():
    seven_hearts = 2 * 13 + 6
    played = [C(seven_hearts)] * 5
    assert _best(played) is HandType.FLUSH_FIVE
    assert played_contains(played, HandType.FLUSH_FIVE) is True
    assert played_contains(played, HandType.FIVE_OF_A_KIND) is False
    assert played_contains(played, HandType.FOUR_OF_A_KIND) is True


def test_five_of_a_kind_contains_four_and_trips_not_flush_five():
    seven_hearts = 2 * 13 + 6
    seven_diamonds = 1 * 13 + 6
    seven_clubs = 0 * 13 + 6
    seven_spades = 3 * 13 + 6
    played = [
        C(seven_hearts),
        C(seven_diamonds),
        C(seven_clubs),
        C(seven_spades),
        Card(seven_spades, int(CardEnhancement.WILD), 0),
    ]
    assert _best(played) is HandType.FIVE_OF_A_KIND
    assert played_contains(played, HandType.FIVE_OF_A_KIND) is True
    assert played_contains(played, HandType.FLUSH_FIVE) is False
    assert played_contains(played, HandType.FOUR_OF_A_KIND) is True


def test_flush_house_contains_full_house_and_flush_house():
    played = [
        C(card_id_from_suit_rank(0, 2)),
        C(card_id_from_suit_rank(0, 2)),
        C(card_id_from_suit_rank(0, 2)),
        C(card_id_from_suit_rank(0, 11)),
        Card(card_id_from_suit_rank(1, 11), int(CardEnhancement.WILD), 0),
    ]
    assert _best(played) is HandType.FLUSH_HOUSE
    assert played_contains(played, HandType.FLUSH_HOUSE) is True
    assert played_contains(played, HandType.FULL_HOUSE) is True
    assert played_contains(played, HandType.FLUSH) is True


# --- cross-checks ---


def test_recognize_trips_implies_contains_pair_not_four():
    played = [C(0), C(13), C(26)]
    assert recognize_poker_hand(played)[0] is HandType.THREE_OF_A_KIND
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.THREE_OF_A_KIND) is True
    assert played_contains(played, HandType.FOUR_OF_A_KIND) is False


def test_recognize_pair_implies_contains_pair_only():
    played = [C(0), C(13), C(1)]
    assert recognize_poker_hand(played)[0] is HandType.PAIR
    assert played_contains(played, HandType.PAIR) is True
    assert played_contains(played, HandType.THREE_OF_A_KIND) is False


# --- invalid length ---


def test_played_contains_empty_raises():
    with pytest.raises(ValueError, match="1..5"):
        played_contains([], HandType.PAIR)


def test_played_contains_six_cards_raises():
    with pytest.raises(ValueError, match="1..5"):
        played_contains([C(0)] * 6, HandType.PAIR)
