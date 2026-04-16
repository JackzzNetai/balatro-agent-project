"""Joker scoring plumbing (M1): activation map, ``try_applying``, stubs do not change totals."""

import numpy as np
import pytest

from defs import (
    CardEnhancement,
    HandType,
    JOKER_ACTIVATION,
    JokerActivation,
    JokerId,
    NO_BOSS_BLIND_ID,
    NUM_JOKERS,
)
from engine import Card, GameSnapshot, Joker
from joker_effects import JokerEffectContext, try_applying_joker_effect
from scoring import ScoreAccumulator, score_play

_NON_SCORING: frozenset[JokerId] = frozenset(
    {
        JokerId.FOUR_FINGERS,
        JokerId.PAREIDOLIA,
        JokerId.SPLASH,
        JokerId.SHORTCUT,
        JokerId.SMEARED_JOKER,
        JokerId.BLUEPRINT,
        JokerId.BRAINSTORM,
    }
)


def _snap(hand_levels: dict, jokers: list[Joker], hand: list[Card] | None = None) -> GameSnapshot:
    return GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=hand or [],
        deck=[],
        jokers=jokers,
        play_remaining=1,
        discard_remaining=0,
        player_hand_size=1,
        hand_levels=hand_levels,
    )


def test_joker_activation_dense_and_passives():
    assert len(JOKER_ACTIVATION) == NUM_JOKERS
    assert set(JOKER_ACTIVATION) == set(JokerId)
    for j in _NON_SCORING:
        assert JOKER_ACTIVATION[j] is None, j
    assert any(JOKER_ACTIVATION[j] is JokerActivation.ON_SCORED for j in JokerId)
    assert any(JOKER_ACTIVATION[j] is JokerActivation.ON_HELD for j in JokerId)
    assert any(JOKER_ACTIVATION[j] is JokerActivation.INDEPENDENT for j in JokerId)


def test_score_play_same_with_passive_and_stub_jokers():
    """Passive jokers + M1 stubs must not alter ``score_play`` totals."""
    played = [Card(0, CardEnhancement.BONUS, 0)]
    levels = {int(HandType.HIGH_CARD): [5, 0]}
    rng = np.random.default_rng(0)
    empty = _snap(levels, [])
    with_jokers = _snap(
        levels,
        [
            Joker(int(JokerId.JOLLY_JOKER), 0),
            Joker(int(JokerId.FOUR_FINGERS), 0),
            Joker(int(JokerId.RAISED_FIST), 0),
        ],
    )
    assert score_play(played, empty, rng) == score_play(played, with_jokers, rng)


def test_independent_passes_played_cards_in_context():
    """Independent handlers receive ``ctx.played`` (the same list as ``score_play`` input)."""
    import joker_effects

    captured: list[list[Card]] = []

    def capture(ctx: JokerEffectContext) -> None:
        captured.append(ctx.played)

    jid = JokerId.JOLLY_JOKER
    old = joker_effects.EFFECT_HANDLERS[jid]
    joker_effects.EFFECT_HANDLERS[jid] = capture
    try:
        played = [Card(0, CardEnhancement.NONE, 0)]
        snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [Joker(int(jid), 0)])
        score_play(played, snap, np.random.default_rng(0))
    finally:
        joker_effects.EFFECT_HANDLERS[jid] = old
    assert captured == [played]


def test_try_applying_skips_when_activation_none():
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    acc = ScoreAccumulator()
    ctx = JokerEffectContext(
        acc=acc,
        snapshot=snap,
        rng=np.random.default_rng(0),
        played=[],
        scored_cards=[],
        scored_card=None,
        held_card=None,
    )
    try_applying_joker_effect(
        JokerActivation.ON_SCORED,
        Joker(int(JokerId.FOUR_FINGERS), 0),
        ctx=ctx,
    )
    assert acc.chips == 0 and acc.mult == 1.0


def test_try_applying_skips_wrong_phase():
    import joker_effects

    called: list[int] = []

    def spy(ctx: JokerEffectContext) -> None:
        called.append(1)

    jid = JokerId.JOLLY_JOKER
    old = joker_effects.EFFECT_HANDLERS[jid]
    joker_effects.EFFECT_HANDLERS[jid] = spy
    try:
        snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
        acc = ScoreAccumulator()
        ctx = JokerEffectContext(
            acc=acc,
            snapshot=snap,
            rng=np.random.default_rng(0),
            played=[],
            scored_cards=[],
            scored_card=None,
            held_card=None,
        )
        try_applying_joker_effect(
            JokerActivation.ON_SCORED,
            Joker(int(jid), 0),
            ctx=ctx,
        )
        assert called == []
        try_applying_joker_effect(
            JokerActivation.INDEPENDENT,
            Joker(int(jid), 0),
            ctx=ctx,
        )
        assert called == [1]
    finally:
        joker_effects.EFFECT_HANDLERS[jid] = old


def test_try_applying_rejects_none_curr_activation():
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    ctx = JokerEffectContext(
        acc=ScoreAccumulator(),
        snapshot=snap,
        rng=np.random.default_rng(0),
        played=[],
        scored_cards=[],
    )
    with pytest.raises(TypeError, match="curr_activation"):
        try_applying_joker_effect(
            None,  # type: ignore[arg-type]
            Joker(0, 0),
            ctx=ctx,
        )


def test_try_applying_invalid_joker_id_raises():
    snap = _snap({int(HandType.HIGH_CARD): [0, 0]}, [])
    ctx = JokerEffectContext(
        acc=ScoreAccumulator(),
        snapshot=snap,
        rng=np.random.default_rng(0),
        played=[],
        scored_cards=[],
    )
    with pytest.raises(ValueError, match="joker.id"):
        try_applying_joker_effect(
            JokerActivation.INDEPENDENT,
            Joker(999, 0),
            ctx=ctx,
        )
