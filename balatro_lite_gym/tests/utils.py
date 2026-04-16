"""Test helpers."""

from defs import NO_BOSS_BLIND_ID
from engine import GameSnapshot


def minimal_snapshot(**overrides) -> GameSnapshot:
    defaults: dict = {
        "target_score": 0,
        "current_score": 0,
        "blind_id": NO_BOSS_BLIND_ID,
        "hand": [],
        "deck": [],
        "jokers": [],
        "play_remaining": 0,
        "discard_remaining": 0,
        "player_hand_size": 0,
        "hand_levels": {},
    }
    defaults.update(overrides)
    return GameSnapshot(**defaults)
