"""Tests for ``util.recognize_poker_hand`` (Balatro wiki, vanilla, no Four Fingers)."""

import pytest

from defs import HandType
from engine import Card
from util import card_rank, recognize_poker_hand


def C(card_id: int, enhancement: int = 0) -> Card:
    return Card(card_id, enhancement, 0)


@pytest.mark.parametrize(
    "played,expected,scored_len",
    [
        ([C(0)], HandType.HIGH_CARD, 1),
        ([C(0), C(13)], HandType.PAIR, 2),
        ([C(0), C(13), C(26)], HandType.THREE_OF_A_KIND, 3),
        ([C(0), C(13), C(1)], HandType.PAIR, 2),
        ([C(41), C(42), C(16), C(3)], HandType.THREE_OF_A_KIND, 3),
    ],
)
def test_high_pair_trips_small(played, expected, scored_len):
    hand, scored = recognize_poker_hand(played)
    assert hand == expected
    assert len(scored) == scored_len


def test_two_pair_four_cards():
    played = [C(0), C(13), C(1), C(14)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.TWO_PAIR
    assert scored == played


def test_two_pair_five_cards():
    played = [C(0), C(13), C(1), C(14), C(2)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.TWO_PAIR
    assert scored == played[:4]


def test_full_house():
    played = [C(0), C(13), C(26), C(1), C(14)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FULL_HOUSE
    assert scored == played


def test_four_of_a_kind_five_cards():
    played = [C(0), C(13), C(26), C(39), C(2)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FOUR_OF_A_KIND
    assert scored == played[:4]


def test_four_of_a_kind_four_cards():
    played = [C(0), C(13), C(26), C(39)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FOUR_OF_A_KIND
    assert scored == played


def test_wheel_straight_mixed_suits():
    played = [
        Card(0, 0, 0),
        Card(1 + 13, 0, 0),
        Card(2 + 26, 0, 0),
        Card(3 + 39, 0, 0),
        Card(4, 0, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.STRAIGHT
    assert scored == played


def test_broadway_straight_mixed_suits():
    played = [
        Card(9, 0, 0),
        Card(10 + 13, 0, 0),
        Card(11 + 26, 0, 0),
        Card(12 + 39, 0, 0),
        Card(0, 0, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.STRAIGHT
    assert scored == played


def test_invalid_wrap_not_straight():
    played = [
        Card(12, 0, 0),
        Card(0 + 13, 0, 0),
        Card(1 + 26, 0, 0),
        Card(2 + 39, 0, 0),
        Card(3, 0, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert scored == [played[1]]


def test_flush_not_straight():
    played = [Card(r, 0, 0) for r in (0, 2, 4, 6, 8)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FLUSH
    assert scored == played


def test_straight_flush_including_royal():
    played = [Card(8 + i, 0, 0) for i in range(5)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.STRAIGHT_FLUSH
    assert scored == played


def test_royal_flush_same_as_straight_flush():
    played = [
        Card(0, 0, 0),
        Card(9, 0, 0),
        Card(10, 0, 0),
        Card(11, 0, 0),
        Card(12, 0, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.STRAIGHT_FLUSH
    assert scored == played


def test_five_of_a_kind_not_all_same_suit():
    played = [Card(0, 0, 0), Card(13, 0, 0), Card(26, 0, 0), Card(39, 0, 0), Card(0, 0, 0)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FIVE_OF_A_KIND
    assert scored == played


def test_flush_five_all_same_rank_same_suit():
    played = [Card(0, 0, 0) for _ in range(5)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FLUSH_FIVE
    assert scored == played


def test_flush_house():
    played = [
        Card(0, 0, 0),
        Card(0, 0, 0),
        Card(0, 0, 0),
        Card(12, 0, 0),
        Card(12, 0, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FLUSH_HOUSE
    assert scored == played


def test_three_club_eights_two_club_tens_is_flush_house():
    """3×♣8 + 2×♣10: full house on ranks, all one suit → Flush House (not plain Full House)."""
    eight_clubs = 0 * 13 + 7
    ten_clubs = 0 * 13 + 9
    played = [Card(eight_clubs, 0, 0)] * 3 + [Card(ten_clubs, 0, 0)] * 2
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FLUSH_HOUSE
    assert scored == played


def test_five_wilds_different_ranks_is_flush_not_five_of_a_kind():
    """All Wild, distinct ranks: flush-feasible, not same rank → FLUSH (not Flush Five / 5oak)."""
    W = 3
    played = [
        Card(0 * 13 + 1, W, 0),
        Card(0 * 13 + 3, W, 0),
        Card(0 * 13 + 5, W, 0),
        Card(0 * 13 + 7, W, 0),
        Card(0 * 13 + 9, W, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FLUSH
    assert scored == played


def test_four_of_a_kind_beats_flush_when_both_apply():
    """4×7♥ + 8♥: also a flush, but Four of a Kind is checked first → FOUR_OF_A_KIND."""
    seven_hearts = 2 * 13 + 6
    eight_hearts = 2 * 13 + 7
    played = [Card(seven_hearts, 0, 0)] * 4 + [Card(eight_hearts, 0, 0)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FOUR_OF_A_KIND
    assert scored == played[:4]


def test_three_eights_short_hand_is_three_of_a_kind():
    """3 cards only: skip all 5-card patterns → THREE_OF_A_KIND."""
    eight_spades = 3 * 13 + 7
    eight_clubs = 0 * 13 + 7
    eight_diamonds = 1 * 13 + 7
    played = [Card(eight_spades, 0, 0), Card(eight_clubs, 0, 0), Card(eight_diamonds, 0, 0)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.THREE_OF_A_KIND
    assert scored == played


def test_fake_flush_wild_ignores_native_spade_suit():
    """2♥,4♥,6♥,8♥ + Wild 9♠: anchor hearts; Wild ignored for suit conflict → FLUSH."""
    W = 3
    two_hearts = 2 * 13 + 1
    four_hearts = 2 * 13 + 3
    six_hearts = 2 * 13 + 5
    eight_hearts = 2 * 13 + 7
    nine_spades_wild = 3 * 13 + 8
    played = [
        Card(two_hearts, 0, 0),
        Card(four_hearts, 0, 0),
        Card(six_hearts, 0, 0),
        Card(eight_hearts, 0, 0),
        Card(nine_spades_wild, W, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.FLUSH
    assert scored == played


def test_flush_five_vs_five_of_a_kind_seven_hearts_all_dup_vs_four_suits_plus_wild():
    """Hand A: five 7♥ → FLUSH_FIVE. Hand B: 7♥♦♣♠ + Wild 7♠ → not flush → FIVE_OF_A_KIND."""
    seven_hearts = 2 * 13 + 6
    played_a = [Card(seven_hearts, 0, 0)] * 5
    hand_a, scored_a = recognize_poker_hand(played_a)
    assert hand_a == HandType.FLUSH_FIVE
    assert scored_a == played_a

    seven_diamonds = 1 * 13 + 6
    seven_clubs = 0 * 13 + 6
    seven_spades = 3 * 13 + 6
    played_b = [
        Card(seven_hearts, 0, 0),
        Card(seven_diamonds, 0, 0),
        Card(seven_clubs, 0, 0),
        Card(seven_spades, 0, 0),
        Card(seven_spades, 3, 0),
    ]
    hand_b, scored_b = recognize_poker_hand(played_b)
    assert hand_b == HandType.FIVE_OF_A_KIND
    assert scored_b == played_b


def test_wild_allows_flush_with_off_suit_card():
    played = [
        Card(2, 0, 0),
        Card(3, 0, 0),
        Card(4, 0, 0),
        Card(5, 0, 0),
        Card(19, 3, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.STRAIGHT_FLUSH
    assert scored == played


def _five_high_card_distinct_ranks_ace_high():
    """Ranks 2,4,6,8,A — mixed suits, not a straight or flush."""
    return [
        Card(0 * 13 + 1, 0, 0),
        Card(1 * 13 + 3, 0, 0),
        Card(2 * 13 + 5, 0, 0),
        Card(3 * 13 + 7, 0, 0),
        Card(0 * 13 + 0, 0, 0),
    ]


def test_high_card_single_card():
    played = [C(0)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert scored == played
    assert scored[0] is played[0]


def test_high_card_five_played_only_ace_scores():
    played = _five_high_card_distinct_ranks_ace_high()
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert len(scored) == 1
    assert scored[0] is played[-1]
    assert card_rank(scored[0].card_id) == 0


def test_high_card_winning_card_can_be_first_middle_or_last():
    base = _five_high_card_distinct_ranks_ace_high()
    ace = base[-1]
    others = base[:-1]
    for pos in (0, 2, 4):
        perm = others[:]
        perm.insert(pos, ace)
        hand, scored = recognize_poker_hand(perm)
        assert hand == HandType.HIGH_CARD
        assert scored == [perm[pos]]
        assert scored[0] is ace


def test_high_card_two_cards_returns_higher_only():
    """Ace high vs lower rank: only the Ace scores."""
    six_diamonds = 1 * 13 + 5
    played = [Card(six_diamonds, 0, 0), C(0)]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert scored == [played[1]]


def test_high_card_five_with_wild_kickers_only_ace_scores():
    """Three naturals in different suits so 2 Wilds cannot make a flush; Ace still sole scorer."""
    W = 3
    played = [
        Card(0 * 13 + 1, 0, 0),
        Card(1 * 13 + 4, 0, 0),
        Card(2 * 13 + 6, W, 0),
        Card(3 * 13 + 8, W, 0),
        Card(3 * 13 + 0, 0, 0),
    ]
    hand, scored = recognize_poker_hand(played)
    assert hand == HandType.HIGH_CARD
    assert scored == [played[-1]]


def test_empty_and_six_cards_raise():
    with pytest.raises(ValueError, match="1..5"):
        recognize_poker_hand([])
    with pytest.raises(ValueError, match="1..5"):
        recognize_poker_hand([C(0)] * 6)


def test_four_fingers_raises():
    with pytest.raises(NotImplementedError):
        recognize_poker_hand([C(0)], four_fingers=True)
