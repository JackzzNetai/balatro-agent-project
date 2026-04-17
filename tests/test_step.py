import math

import numpy as np
import pytest

from environment import BalatroEnv, MAX_HAND_LENGTH
from engine import Card, GameSnapshot
from defs import HandType, NO_BOSS_BLIND_ID
from utils import minimal_snapshot


def _zeros_selection():
    return np.zeros(MAX_HAND_LENGTH, dtype=np.int8)


def _select_indices(*indices: int):
    s = np.zeros(MAX_HAND_LENGTH, dtype=np.int8)
    for i in indices:
        s[i] = 1
    return s


def test_invalid_selection_no_op_reward_minus_one_terminated_false():
    snap = GameSnapshot(
        target_score=0,
        current_score=100,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0), Card(1, 0, 0)],
        deck=[],
        jokers=[],
        play_remaining=0,
        discard_remaining=1,
        player_hand_size=2,
        hand_levels={},
    )
    env = BalatroEnv(snap)
    env.reset(seed=0)
    before = (env._snapshot.current_score, len(env._snapshot.hand))
    obs, r, term, trunc, info = env.step(
        {"selection": _zeros_selection(), "action_type": 0}
    )
    assert r == -1
    assert term is False
    assert trunc is False
    assert info["snapshot"] is env._snapshot
    assert (env._snapshot.current_score, len(env._snapshot.hand)) == before
    assert env.observation_space.contains(obs)


def test_valid_discard_removes_and_draws():
    hand = [Card(0, 0, 0), Card(1, 0, 0), Card(2, 0, 0)]
    deck = [Card(10, 0, 0), Card(11, 0, 0)]
    snap = GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=hand,
        deck=deck,
        jokers=[],
        play_remaining=5,
        discard_remaining=3,
        player_hand_size=3,
        hand_levels={},
    )
    env = BalatroEnv(snap)
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(
        {"selection": _select_indices(0), "action_type": 0}
    )
    assert r == 0
    assert trunc is False
    assert info["snapshot"] is env._snapshot
    assert env._snapshot.discard_remaining == 2
    assert len(env._snapshot.hand) == 3
    assert len(env._snapshot.deck) == 1
    assert env.observation_space.contains(obs)


def test_valid_play_terminated_skips_draw():
    hand = [Card(i, 0, 0) for i in range(5)]
    deck = [Card(20, 0, 0), Card(21, 0, 0), Card(22, 0, 0)]
    snap = GameSnapshot(
        target_score=999,
        current_score=100,
        blind_id=NO_BOSS_BLIND_ID,
        hand=hand,
        deck=deck,
        jokers=[],
        play_remaining=1,
        discard_remaining=2,
        player_hand_size=5,
        hand_levels={int(HandType.HIGH_CARD): 1},
    )
    env = BalatroEnv(snap)
    env.reset(seed=2)
    obs, r, term, trunc, info = env.step(
        {"selection": _select_indices(0), "action_type": 1}
    )
    # Played Card(0,0,0): Plain (NONE); Ace rank chips 11 + hand line chips 5 → 16; mult 1
    assert env._snapshot.current_score == 116
    assert r == pytest.approx(0.0 + math.sqrt(math.log10(116.0)))
    assert term is True
    assert trunc is False
    assert info["snapshot"] is env._snapshot
    assert env._snapshot.play_remaining == 0
    assert len(env._snapshot.hand) == 4
    assert len(env._snapshot.deck) == 3
    assert env.observation_space.contains(obs)


def test_terminated_by_score_reward_uses_play_remaining_and_log_score():
    class BoomEnv(BalatroEnv):
        def __init__(self, snapshot: GameSnapshot):
            super().__init__(snapshot)

        def _calculate_score(self, selected_cards):
            return 50

    snap = GameSnapshot(
        target_score=100,
        current_score=60,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0)],
        deck=[],
        jokers=[],
        play_remaining=3,
        discard_remaining=1,
        player_hand_size=1,
        hand_levels={},
    )
    env2 = BoomEnv(snap)
    env2.reset(seed=0)
    obs, r, term, trunc, info = env2.step(
        {"selection": _select_indices(0), "action_type": 1}
    )
    assert term is True
    assert env2._snapshot.current_score == 110
    assert env2._snapshot.play_remaining == 2
    expected = 2.0 + math.sqrt(math.log10(110.0))
    assert r == pytest.approx(expected)
    assert info["snapshot"] is env2._snapshot
    assert env2.observation_space.contains(obs)


def test_invalid_action_type_raises():
    snap = GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0)],
        deck=[],
        jokers=[],
        play_remaining=5,
        discard_remaining=1,
        player_hand_size=1,
        hand_levels={},
    )
    env = BalatroEnv(snap)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="invalid action_type"):
        env.step({"selection": _select_indices(0), "action_type": 2})
    assert len(env._snapshot.hand) == 1


def test_discard_remaining_zero_no_op_like_invalid_selection():
    snap = GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0), Card(1, 0, 0)],
        deck=[],
        jokers=[],
        play_remaining=3,
        discard_remaining=0,
        player_hand_size=2,
        hand_levels={},
    )
    env = BalatroEnv(snap)
    env.reset(seed=0)
    before = (len(env._snapshot.hand), env._snapshot.discard_remaining)
    obs, r, term, trunc, info = env.step(
        {"selection": _select_indices(0), "action_type": 0}
    )
    assert r == -1
    assert term is False
    assert info["snapshot"] is env._snapshot
    assert (len(env._snapshot.hand), env._snapshot.discard_remaining) == before
    assert env.observation_space.contains(obs)


def test_play_remaining_zero_raises():
    snap = GameSnapshot(
        target_score=999,
        current_score=50,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0)],
        deck=[],
        jokers=[],
        play_remaining=0,
        discard_remaining=2,
        player_hand_size=1,
        hand_levels={},
    )
    env = BalatroEnv(snap)
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="play_remaining is 0"):
        env.step({"selection": _select_indices(0), "action_type": 1})
    assert len(env._snapshot.hand) == 1
