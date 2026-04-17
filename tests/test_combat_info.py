"""``BalatroEnv`` info: ``combat_won`` on terminal steps."""

import pytest

from defs import HandType, NO_BOSS_BLIND_ID
from engine import Card, GameSnapshot
from environment import BalatroEnv, MAX_HAND_LENGTH
from utils import minimal_snapshot


def _select_indices(*indices: int):
    s = [0] * MAX_HAND_LENGTH
    for i in indices:
        s[i] = 1
    return s


def test_reset_info_snapshot_only():
    snap = minimal_snapshot(
        hand=[Card(0, 0, 0)],
        play_remaining=1,
        discard_remaining=1,
        player_hand_size=1,
    )
    env = BalatroEnv(snap)
    _obs, info = env.reset(seed=0)
    assert "combat_won" not in info
    assert "snapshot" in info


def test_non_terminal_step_info_snapshot_only():
    snap = GameSnapshot(
        target_score=999,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0), Card(1, 0, 0)],
        deck=[Card(10, 0, 0)],
        jokers=[],
        play_remaining=5,
        discard_remaining=3,
        player_hand_size=2,
        hand_levels={},
    )
    env = BalatroEnv(snap)
    env.reset(seed=1)
    _obs, _r, term, _trunc, info = env.step(
        {"selection": _select_indices(0), "action_type": 0}
    )
    assert term is False
    assert "combat_won" not in info


def test_terminal_win_combat_flags_true():
    class BoomEnv(BalatroEnv):
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
    env = BoomEnv(snap)
    env.reset(seed=0)
    _obs, _r, term, _trunc, info = env.step(
        {"selection": _select_indices(0), "action_type": 1}
    )
    assert term is True
    assert env._snapshot.current_score >= env._snapshot.target_score
    assert info["combat_won"] is True


def test_terminal_loss_plays_exhausted_combat_flags_false():
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
        hand_levels={int(HandType.HIGH_CARD): [5, 0]},
    )
    env = BalatroEnv(snap)
    env.reset(seed=2)
    _obs, _r, term, _trunc, info = env.step(
        {"selection": _select_indices(0), "action_type": 1}
    )
    assert term is True
    assert env._snapshot.current_score < env._snapshot.target_score
    assert info["combat_won"] is False
