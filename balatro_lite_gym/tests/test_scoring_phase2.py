"""On-scored path for ``score_play``: hand_levels, rank chips, enhancements, editions."""

import numpy as np
import pytest

from defs import CardEnhancement, Edition, HandType, NO_BOSS_BLIND_ID
from engine import Card, GameSnapshot
from scoring import rank_chips, score_play


def _snap(hand_levels: dict, hand: list[Card] | None = None) -> GameSnapshot:
    return GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=hand or [],
        deck=[],
        jokers=[],
        play_remaining=1,
        discard_remaining=0,
        player_hand_size=1,
        hand_levels=hand_levels,
    )


@pytest.mark.parametrize(
    "card_rank,expected",
    [
        (0, 11),
        (1, 2),
        (8, 9),
        (9, 10),
        (10, 10),
        (12, 10),
    ],
)
def test_rank_chips_table(card_rank, expected):
    assert rank_chips(card_rank) == expected


def test_score_play_high_card_single_ace_bonus_base():
    """Hand line + rank + Bonus on scored Ace only."""
    played = [Card(0, CardEnhancement.BONUS, 0)]
    snap = _snap({int(HandType.HIGH_CARD): [5, 0]})
    rng = np.random.default_rng(0)
    assert score_play(played, snap, rng) == 5 + 11 + 30


def test_score_play_pair_two_aces_hand_mult():
    played = [
        Card(0, CardEnhancement.BONUS, 0),
        Card(13, CardEnhancement.MULT, 0),
    ]
    snap = _snap({int(HandType.PAIR): [10, 2]})
    rng = np.random.default_rng(0)
    # hand: chips +10, mult +2 → 3; pair scores both aces: 11+30 + 11+4 chips/mult parts
    assert score_play(played, snap, rng) == int((10 + 11 + 30 + 11) * (1.0 + 2.0 + 4.0))


def test_missing_hand_levels_key_raises():
    played = [Card(0, 0, 0)]
    snap = _snap({})
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError, match="hand_levels missing"):
        score_play(played, snap, rng)


def test_score_play_rng_none_raises():
    played = [Card(0, 0, 0)]
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]})
    with pytest.raises(TypeError, match="requires"):
        score_play(played, snap, None)  # type: ignore[arg-type]


def test_editions_foil_holo_poly_order():
    played = [Card(0, CardEnhancement.BONUS, int(Edition.FOIL))]
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]})
    rng = np.random.default_rng(0)
    # rank 11 + bonus 30 + foil 50 = 91, mult 1
    assert score_play(played, snap, rng) == 91

    # enhancement Wild (no chip/mult); Holo +10 mult on rank chips only
    played2 = [Card(0, CardEnhancement.WILD, int(Edition.HOLOGRAPHIC))]
    assert score_play(played2, snap, rng) == int(11.0 * 11.0)

    played3 = [Card(0, CardEnhancement.WILD, int(Edition.POLYCHROME))]
    assert score_play(played3, snap, rng) == int(11 * 1.5)


def test_lucky_mult_triggers_with_low_rng():
    class _Rng:
        def random(self) -> float:
            return 0.0

    played = [Card(0, CardEnhancement.LUCKY, 0)]
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]})
    # chips 11, mult 1 + 20 = 21
    assert score_play(played, snap, _Rng()) == int(11 * 21.0)


def test_lucky_mult_not_triggered_with_high_rng():
    class _Rng:
        def random(self) -> float:
            return 0.99

    played = [Card(0, CardEnhancement.LUCKY, 0)]
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]})
    assert score_play(played, snap, _Rng()) == 11


def test_hand_levels_pair_length_invalid():
    played = [Card(0, 0, 0)]
    snap = GameSnapshot(
        target_score=0,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[],
        deck=[],
        jokers=[],
        play_remaining=0,
        discard_remaining=0,
        player_hand_size=0,
        hand_levels={int(HandType.HIGH_CARD): [1]},
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="length-2"):
        score_play(played, snap, rng)
