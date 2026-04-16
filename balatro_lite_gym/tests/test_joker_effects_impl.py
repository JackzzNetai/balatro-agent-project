"""Behavior tests for implemented jokers in ``joker_effects`` / ``score_play``.

Each test targets a small, deterministic scenario. Stochastic jokers use fixed RNG seeds.
"""

from __future__ import annotations

import numpy as np

from defs import CardEnhancement, HandType, JokerActivation, JokerId, NO_BOSS_BLIND_ID
from engine import Card, GameSnapshot, Joker
from joker_effects import JokerEffectContext, try_applying_joker_effect
from scoring import ScoreAccumulator, score_play
from util import card_id_from_suit_rank


def _snap(
    hand_levels: dict[int, list[int]],
    jokers: list[Joker],
    *,
    hand: list[Card] | None = None,
    deck: list[Card] | None = None,
    play_remaining: int = 1,
    discard_remaining: int = 0,
) -> GameSnapshot:
    return GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=list(hand or []),
        deck=list(deck or []),
        jokers=jokers,
        play_remaining=play_remaining,
        discard_remaining=discard_remaining,
        player_hand_size=1,
        hand_levels=hand_levels,
    )


def _cid(suit: int, rank: int) -> int:
    return card_id_from_suit_rank(suit, rank)


# --- Independent ---


def test_joker_independent_plus_four_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(3, 0), CardEnhancement.NONE, 0)]  # Ace spades
    base = score_play(played, _snap(levels, []), np.random.default_rng(0))
    with_j = score_play(
        played,
        _snap(levels, [Joker(int(JokerId.JOKER), 0)]),
        np.random.default_rng(0),
    )
    assert base == 11  # rank chips only
    assert with_j == 55  # 11 * (1 + 4)


def test_jolly_joker_pair_adds_eight_mult():
    levels = {int(HandType.PAIR): [0, 0]}
    played = [
        Card(_cid(0, 12), CardEnhancement.NONE, 0),  # K♣
        Card(_cid(1, 12), CardEnhancement.NONE, 0),  # K♦
    ]
    s = _snap(levels, [Joker(int(JokerId.JOLLY_JOKER), 0)])
    assert score_play(played, s, np.random.default_rng(0)) == 180  # 20 chips * 9 mult


def test_half_joker_three_or_fewer_cards():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]  # A♣
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.HALF_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 231
    )  # 11 * (1 + 20)


def test_banner_discard_remaining_chips():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.BANNER), 0)], discard_remaining=2),
            np.random.default_rng(0),
        )
        == 71
    )  # (11 + 60) * 1


def test_mystic_summit_zero_discards():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.MYSTIC_SUMMIT), 0)], discard_remaining=0),
            np.random.default_rng(0),
        )
        == 176
    )  # 11 * (1 + 15)


def test_blue_joker_deck_chips():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    deck = [Card(i, 0, 0) for i in range(5)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.BLUE_JOKER), 0)], deck=deck),
            np.random.default_rng(0),
        )
        == 21
    )  # (11 + 10) * 1


def test_blackboard_empty_hand_triple_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    assert (
        score_play(played, _snap(levels, [Joker(int(JokerId.BLACKBOARD), 0)], hand=[]), np.random.default_rng(0))
        == 33
    )  # 11 * 3


def test_flower_pot_four_suits_multiplies():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [
        Card(_cid(0, 4), CardEnhancement.NONE, 0),  # 5♣
        Card(_cid(1, 3), CardEnhancement.NONE, 0),  # 4♦
        Card(_cid(2, 2), CardEnhancement.NONE, 0),  # 3♥
        Card(_cid(3, 1), CardEnhancement.NONE, 0),  # 2♠
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.FLOWER_POT), 0)]),
            np.random.default_rng(0),
        )
        == 15
    )  # highest scores alone: 5♣ → 5 chips * 3


def test_seeing_double_club_and_non_club_pair():
    levels = {int(HandType.PAIR): [0, 0]}
    played = [
        Card(_cid(0, 0), CardEnhancement.NONE, 0),  # A♣
        Card(_cid(2, 0), CardEnhancement.NONE, 0),  # A♥
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.SEEING_DOUBLE), 0)]),
            np.random.default_rng(0),
        )
        == 44
    )  # 22 chips * 2


def test_the_duo_pair_doubles_mult():
    levels = {int(HandType.PAIR): [0, 0]}
    played = [
        Card(_cid(0, 12), CardEnhancement.NONE, 0),
        Card(_cid(1, 12), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.THE_DUO), 0)]),
            np.random.default_rng(0),
        )
        == 40
    )  # 20 * 2


def test_the_trio_three_of_a_kind_triples_mult():
    levels = {int(HandType.THREE_OF_A_KIND): [0, 0]}
    played = [
        Card(_cid(0, 0), CardEnhancement.NONE, 0),
        Card(_cid(1, 0), CardEnhancement.NONE, 0),
        Card(_cid(2, 0), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.THE_TRIO), 0)]),
            np.random.default_rng(0),
        )
        == 99
    )  # 33 chips * 3


def test_sly_joker_pair_adds_fifty_chips():
    levels = {int(HandType.PAIR): [0, 0]}
    played = [
        Card(_cid(0, 12), CardEnhancement.NONE, 0),
        Card(_cid(1, 12), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.SLY_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 70
    )  # (20 + 50) * 1


def test_zany_joker_three_of_a_kind_adds_twelve_mult():
    levels = {int(HandType.THREE_OF_A_KIND): [0, 0]}
    played = [
        Card(_cid(0, 5), CardEnhancement.NONE, 0),
        Card(_cid(1, 5), CardEnhancement.NONE, 0),
        Card(_cid(2, 5), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.ZANY_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 234
    )  # 18 chips * 13 mult


def test_acrobat_final_play_triples_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.ACROBAT), 0)], play_remaining=0),
            np.random.default_rng(0),
        )
        == 33
    )  # 11 * 3


def test_misprint_independent_adds_rng_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    # ``np.random.default_rng(21)`` first ``integers(0, 24)`` is 7 (Misprint INDEPENDENT).
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.MISPRINT), 0)]),
            np.random.default_rng(21),
        )
        == 88
    )  # 11 * (1 + 7)


# --- On held ---


def test_raised_fist_only_first_held_and_rank_chips_double():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    two_clubs = Card(_cid(0, 1), CardEnhancement.NONE, 0)  # 2♣
    queen = Card(_cid(0, 11), CardEnhancement.NONE, 0)  # Q♣
    king_played = Card(_cid(1, 12), CardEnhancement.NONE, 0)  # K♦
    played = [king_played]
    hand = [two_clubs, queen]
    # Lowest rank is 2 (rank index 1) → rank_chips(1)=2 → +4 Mult once from Raised Fist.
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.RAISED_FIST), 0)], hand=hand),
            np.random.default_rng(0),
        )
        == 50
    )  # 10 chips * (1 + 4) mult; only ``hand[0]`` triggers Raised Fist


def test_shoot_the_moon_queen_held():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(3, 0), CardEnhancement.NONE, 0)]  # A♠
    q = Card(_cid(0, 11), CardEnhancement.NONE, 0)
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.SHOOT_THE_MOON), 0)], hand=[q]),
            np.random.default_rng(0),
        )
        == 154
    )  # 11 * (1 + 13)


def test_baron_each_king_multiplies():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(2, 0), CardEnhancement.NONE, 0)]  # A♥
    k1, k2 = Card(_cid(0, 12), CardEnhancement.NONE, 0), Card(_cid(1, 12), CardEnhancement.NONE, 0)
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.BARON), 0)], hand=[k1, k2]),
            np.random.default_rng(0),
        )
        == 24
    )  # 11 * 1.5 * 1.5


# --- More independent pattern jokers ---


def test_mad_joker_two_pair():
    levels = {int(HandType.TWO_PAIR): [0, 0]}
    played = [
        Card(_cid(0, 12), CardEnhancement.NONE, 0),
        Card(_cid(1, 12), CardEnhancement.NONE, 0),
        Card(_cid(0, 8), CardEnhancement.NONE, 0),
        Card(_cid(1, 8), CardEnhancement.NONE, 0),
        Card(_cid(2, 1), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.MAD_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 418
    )  # 38 chips * 11 mult


def test_crazy_joker_straight_adds_twelve_mult():
    levels = {int(HandType.STRAIGHT): [0, 0]}
    played = [
        Card(_cid(0, 0), CardEnhancement.NONE, 0),
        Card(_cid(1, 1), CardEnhancement.NONE, 0),
        Card(_cid(2, 2), CardEnhancement.NONE, 0),
        Card(_cid(3, 3), CardEnhancement.NONE, 0),
        Card(_cid(0, 4), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.CRAZY_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 325
    )  # 25 chips * 13 mult


def test_droll_joker_flush_adds_ten_mult():
    levels = {int(HandType.FLUSH): [0, 0]}
    played = [Card(_cid(3, r), CardEnhancement.NONE, 0) for r in (1, 3, 5, 8, 10)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.DROLL_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 341
    )  # 31 chips * 11 mult


def test_clever_joker_two_pair_chips():
    levels = {int(HandType.TWO_PAIR): [0, 0]}
    played = [
        Card(_cid(0, 12), CardEnhancement.NONE, 0),
        Card(_cid(1, 12), CardEnhancement.NONE, 0),
        Card(_cid(0, 8), CardEnhancement.NONE, 0),
        Card(_cid(1, 8), CardEnhancement.NONE, 0),
        Card(_cid(2, 1), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.CLEVER_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 118
    )  # (38 + 80) * 1


def test_devious_joker_straight_chips():
    levels = {int(HandType.STRAIGHT): [0, 0]}
    played = [
        Card(_cid(0, 0), CardEnhancement.NONE, 0),
        Card(_cid(1, 1), CardEnhancement.NONE, 0),
        Card(_cid(2, 2), CardEnhancement.NONE, 0),
        Card(_cid(3, 3), CardEnhancement.NONE, 0),
        Card(_cid(0, 4), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.DEVIOUS_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 125
    )  # (25 + 100) * 1


def test_crafty_joker_flush_chips():
    levels = {int(HandType.FLUSH): [0, 0]}
    played = [Card(_cid(3, r), CardEnhancement.NONE, 0) for r in (1, 3, 5, 8, 10)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.CRAFTY_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 111
    )  # (31 + 80) * 1


def test_wily_joker_three_of_a_kind_chips():
    levels = {int(HandType.THREE_OF_A_KIND): [0, 0]}
    played = [
        Card(_cid(0, 6), CardEnhancement.NONE, 0),
        Card(_cid(1, 6), CardEnhancement.NONE, 0),
        Card(_cid(2, 6), CardEnhancement.NONE, 0),
        Card(_cid(3, 1), CardEnhancement.NONE, 0),
        Card(_cid(0, 2), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.WILY_JOKER), 0)]),
            np.random.default_rng(0),
        )
        == 121
    )  # (7+7+7 + 100) * 1


def test_the_order_straight_triples_mult():
    levels = {int(HandType.STRAIGHT): [0, 0]}
    played = [
        Card(_cid(0, 0), CardEnhancement.NONE, 0),
        Card(_cid(1, 1), CardEnhancement.NONE, 0),
        Card(_cid(2, 2), CardEnhancement.NONE, 0),
        Card(_cid(3, 3), CardEnhancement.NONE, 0),
        Card(_cid(0, 4), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.THE_ORDER), 0)]),
            np.random.default_rng(0),
        )
        == 75
    )  # 25 * 3


def test_the_tribe_flush_doubles_mult():
    levels = {int(HandType.FLUSH): [0, 0]}
    played = [Card(_cid(3, r), CardEnhancement.NONE, 0) for r in (1, 3, 5, 8, 10)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.THE_TRIBE), 0)]),
            np.random.default_rng(0),
        )
        == 62
    )  # 31 * 2


def test_the_family_four_of_a_kind_quadruples_mult():
    levels = {int(HandType.FOUR_OF_A_KIND): [0, 0]}
    played = [
        Card(_cid(0, 12), CardEnhancement.NONE, 0),
        Card(_cid(1, 12), CardEnhancement.NONE, 0),
        Card(_cid(2, 12), CardEnhancement.NONE, 0),
        Card(_cid(3, 12), CardEnhancement.NONE, 0),
        Card(_cid(1, 1), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.THE_FAMILY), 0)]),
            np.random.default_rng(0),
        )
        == 160
    )  # 40 * 4


# --- On scored (direct dispatch) ---


def test_try_applying_greedy_counts_wild_as_diamond():
    acc = ScoreAccumulator()
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    wild = Card(_cid(0, 5), CardEnhancement.WILD, 0)  # printed 6♣, Wild
    ctx = JokerEffectContext(
        acc=acc,
        snapshot=snap,
        played=[wild],
        scored_cards=[wild],
        scored_card=wild,
        rng=np.random.default_rng(0),
    )
    try_applying_joker_effect(
        JokerActivation.ON_SCORED,
        Joker(int(JokerId.GREEDY_JOKER), 0),
        ctx=ctx,
    )
    assert acc.mult == 4.0  # 1 + 3


def test_try_applying_lusty_heart():
    acc = ScoreAccumulator()
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    h = Card(_cid(2, 7), CardEnhancement.NONE, 0)  # 8♥
    ctx = JokerEffectContext(
        acc=acc,
        snapshot=snap,
        played=[h],
        scored_cards=[h],
        scored_card=h,
        rng=np.random.default_rng(0),
    )
    try_applying_joker_effect(
        JokerActivation.ON_SCORED,
        Joker(int(JokerId.LUSTY_JOKER), 0),
        ctx=ctx,
    )
    assert acc.mult == 4.0


def test_try_applying_gluttonous_club():
    acc = ScoreAccumulator()
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    x = Card(_cid(0, 4), CardEnhancement.NONE, 0)  # 5♣
    ctx = JokerEffectContext(
        acc=acc,
        snapshot=snap,
        played=[x],
        scored_cards=[x],
        scored_card=x,
        rng=np.random.default_rng(0),
    )
    try_applying_joker_effect(
        JokerActivation.ON_SCORED,
        Joker(int(JokerId.GLUTTONOUS_JOKER), 0),
        ctx=ctx,
    )
    assert acc.mult == 4.0


def test_try_applying_wrathful_spade():
    acc = ScoreAccumulator()
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    x = Card(_cid(3, 2), CardEnhancement.NONE, 0)  # 3♠
    ctx = JokerEffectContext(
        acc=acc,
        snapshot=snap,
        played=[x],
        scored_cards=[x],
        scored_card=x,
        rng=np.random.default_rng(0),
    )
    try_applying_joker_effect(
        JokerActivation.ON_SCORED,
        Joker(int(JokerId.WRATHFUL_JOKER), 0),
        ctx=ctx,
    )
    assert acc.mult == 4.0


def test_fibonacci_scored_rank_via_score_play():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 2), CardEnhancement.NONE, 0)]  # 3♣ (Fibonacci rank)
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.FIBONACCI), 0)]),
            np.random.default_rng(0),
        )
        == 27
    )  # 3 chips * (1 + 8) mult


def test_scary_face_face_card_chips():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 10), CardEnhancement.NONE, 0)]  # J♣
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.SCARY_FACE), 0)]),
            np.random.default_rng(0),
        )
        == 40
    )  # (10 + 30) * 1


def test_smiley_face_face_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 10), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.SMILEY_FACE), 0)]),
            np.random.default_rng(0),
        )
        == 60
    )  # 10 chips * 6 mult


def test_even_steven_ten_rank():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 9), CardEnhancement.NONE, 0)]  # 10♣
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.EVEN_STEVEN), 0)]),
            np.random.default_rng(0),
        )
        == 50
    )  # 10 * (1 + 4)


def test_odd_todd_nine_rank_chips():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 8), CardEnhancement.NONE, 0)]  # 9♣
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.ODD_TODD), 0)]),
            np.random.default_rng(0),
        )
        == 40
    )  # (9 + 31) * 1


def test_scholar_ace_chips_and_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 0), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.SCHOLAR), 0)]),
            np.random.default_rng(0),
        )
        == 155
    )  # (11 + 20) * 5


def test_walkie_talkie_ten_rank():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 9), CardEnhancement.NONE, 0)]  # 10♣
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.WALKIE_TALKIE), 0)]),
            np.random.default_rng(0),
        )
        == 100
    )  # (10 + 10) chips * (1 + 4) mult


def test_walkie_talkie_pair_of_fours():
    levels = {int(HandType.PAIR): [0, 0]}
    played = [
        Card(_cid(0, 3), CardEnhancement.NONE, 0),
        Card(_cid(1, 3), CardEnhancement.NONE, 0),
    ]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.WALKIE_TALKIE), 0)]),
            np.random.default_rng(0),
        )
        == 252
    )  # (4+4+10+10) chips * (1+4+4) mult


def test_arrowhead_spade_chips():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(3, 11), CardEnhancement.NONE, 0)]  # Q♠
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.ARROWHEAD), 0)]),
            np.random.default_rng(0),
        )
        == 60
    )  # (10 + 50) * 1 — Queen rank uses 10 base chips


def test_onyx_agate_club_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 6), CardEnhancement.NONE, 0)]  # 7♣
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.ONYX_AGATE), 0)]),
            np.random.default_rng(0),
        )
        == 56
    )  # 7 * (1 + 7)


def test_ancient_clubs_multiplicative_per_scoring_club():
    levels = {int(HandType.PAIR): [0, 0]}
    played = [Card(_cid(0, 12), CardEnhancement.NONE, 0), Card(_cid(1, 12), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.ANCIENT_JOKER_CLUBS), 0)]),
            np.random.default_rng(0),
        )
        == 30
    )  # 20 chips, only K♣ triggers ×1.5 → 20 * 1.5 = 30


def test_triboulet_queen_doubles_mult():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 11), CardEnhancement.NONE, 0)]
    assert (
        score_play(
            played,
            _snap(levels, [Joker(int(JokerId.TRIBOULET), 0)]),
            np.random.default_rng(0),
        )
        == 20
    )  # 10 chips * 2 mult


def test_bloodstone_heart_proc_and_no_proc():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(2, 0), CardEnhancement.NONE, 0)]  # A♥
    s = _snap(levels, [Joker(int(JokerId.BLOODSTONE), 0)])
    assert score_play(played, s, np.random.default_rng(2)) == 16  # first random() < 0.5 → ×1.5
    assert score_play(played, s, np.random.default_rng(0)) == 11  # first random() >= 0.5


def test_photograph_is_noop_for_total():
    levels = {int(HandType.HIGH_CARD): [0, 0]}
    played = [Card(_cid(0, 10), CardEnhancement.NONE, 0)]
    base = score_play(played, _snap(levels, []), np.random.default_rng(0))
    with_photo = score_play(
        played,
        _snap(levels, [Joker(int(JokerId.PHOTOGRAPH), 0)]),
        np.random.default_rng(0),
    )
    assert base == with_photo == 10
