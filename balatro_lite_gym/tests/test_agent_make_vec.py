"""Smoke tests for ``agent.lite_combat_env.make_vec`` (sync + async backends)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _tiny_pool():
    from defs import HandType, JokerId
    from engine import Card, GameSnapshot, Joker

    hl = {int(h): [10, 2] for h in HandType}
    snap = GameSnapshot(
        100,
        0,
        -1,
        [Card(0, 0, 0), Card(1, 0, 0)],
        [Card(2, 0, 0)] * 12,
        [Joker(int(JokerId.JOKER), 0)],
        2,
        1,
        8,
        hl,
    )
    return [snap]


def test_make_vec_sync_reset_step():
    from agent.lite_combat_env import make_vec
    from environment import MAX_HAND_LENGTH

    vec = make_vec(_tiny_pool(), n=2, base_seed=0, backend="sync")
    try:
        obs, _ = vec.reset(seed=0)
        assert obs["hand_card_ids"].shape == (2, MAX_HAND_LENGTH)
        a = np.zeros((2, MAX_HAND_LENGTH + 1), dtype=np.int8)
        a[:, 0] = 1
        a[:, 1] = 1
        obs2, r, d, tr, _ = vec.step(a)
        assert obs2["hand_card_ids"].shape == (2, MAX_HAND_LENGTH)
        assert r.shape == (2,)
        assert d.shape == (2,)
    finally:
        vec.close()


def test_make_vec_async_reset_step():
    from agent.lite_combat_env import make_vec
    from environment import MAX_HAND_LENGTH

    vec = make_vec(_tiny_pool(), n=2, base_seed=0, backend="async")
    try:
        obs, _ = vec.reset(seed=0)
        assert obs["hand_card_ids"].shape == (2, MAX_HAND_LENGTH)
        a = np.zeros((2, MAX_HAND_LENGTH + 1), dtype=np.int8)
        a[:, 0] = 1
        a[:, 1] = 1
        obs2, r, d, tr, _ = vec.step(a)
        assert obs2["hand_card_ids"].shape == (2, MAX_HAND_LENGTH)
        assert r.shape == (2,)
        assert d.shape == (2,)
    finally:
        vec.close()
