import numpy as np

from defs import NO_BOSS_BLIND_ID
from engine import Card, GameSnapshot
from environment import BalatroEnv, MAX_HAND_LENGTH
from utils import minimal_snapshot


def _select_indices(*indices: int):
    s = np.zeros(MAX_HAND_LENGTH, dtype=np.int8)
    for i in indices:
        s[i] = 1
    return s


def test_reset_with_snapshot_updates_template_for_next_bare_reset():
    base = minimal_snapshot(
        target_score=10,
        hand=[Card(0, 0, 0)],
        play_remaining=1,
        player_hand_size=1,
    )
    env = BalatroEnv(base)
    env.reset(seed=0)

    other = GameSnapshot(
        target_score=99,
        current_score=0,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(5, 0, 0), Card(6, 0, 0)],
        deck=[],
        jokers=[],
        play_remaining=2,
        discard_remaining=1,
        player_hand_size=2,
        hand_levels={},
    )
    env.reset(seed=1, options={"snapshot": other})
    assert env._snapshot is other
    assert env._init_snapshot_template.target_score == 99
    assert len(env._init_snapshot_template.hand) == 2

    env.reset(seed=2)
    assert env._snapshot is not other
    assert env._snapshot.target_score == 99
    assert len(env._snapshot.hand) == 2


def test_bare_reset_restores_template_not_mutated_live_state():
    snap = minimal_snapshot(
        target_score=999,
        hand=[Card(0, 0, 0), Card(1, 0, 0)],
        deck=[Card(10, 0, 0)],
        play_remaining=5,
        discard_remaining=2,
        player_hand_size=2,
    )
    env = BalatroEnv(snap)
    env.reset(seed=0)
    env.step({"selection": _select_indices(0), "action_type": 0})
    assert env._snapshot.hand[0].card_id != snap.hand[0].card_id

    env.reset(seed=1)
    assert env._snapshot.hand[0].card_id == snap.hand[0].card_id
