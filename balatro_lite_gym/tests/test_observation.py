from defs import BossBlind
from environment import BalatroEnv, snapshot_to_obs_dict
from engine import Card, GameSnapshot, Joker
from utils import minimal_snapshot


def test_reset_observation_is_in_space():
    env = BalatroEnv(minimal_snapshot())
    obs, info = env.reset(seed=0)
    assert info["snapshot"] is env._snapshot
    assert env.observation_space.contains(obs)


def test_snapshot_to_obs_contains_with_sample_cards():
    snap = GameSnapshot(
        target_score=300,
        current_score=50,
        blind_id=BossBlind.THE_HOOK,
        hand=[Card(0, 0, 0), Card(51, 1, 2)],
        deck=[Card(3, 0, 0)],
        jokers=[Joker(5, 0)],
        play_remaining=4,
        discard_remaining=3,
        player_hand_size=5,
        hand_levels={0: [10, 1], 11: [100, 4]},
    )
    obs = snapshot_to_obs_dict(snap)
    env = BalatroEnv(snap)
    assert env.observation_space.contains(obs)
