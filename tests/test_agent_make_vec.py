"""Smoke tests for ``env.lite_combat_env.make_vec`` (sync + async backends)."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


def _tiny_pool():
    from defs import HandType, JokerId
    from engine import Card, GameSnapshot, Joker

    hl = {int(h): 2 for h in HandType}
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
    from env.lite_combat_env import make_vec
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
    from env.lite_combat_env import make_vec
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


def test_import_agent_does_not_load_torch():
    """``import agent`` should not require PyTorch (lazy ``CombatPPOAgent``)."""
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]
    if "torch" in sys.modules:
        del sys.modules["torch"]
    import agent as agent_mod  # noqa: F401

    assert "torch" not in sys.modules


def test_combat_ppo_forward_smoke():
    torch = pytest.importorskip("torch")
    from env.lite_combat_env import dict_to_tensors, make_vec
    from agent.model import CombatPPOAgent
    from environment import MAX_HAND_LENGTH

    vec = make_vec(_tiny_pool(), n=3, base_seed=0, backend="sync")
    try:
        obs, _ = vec.reset(seed=0)
        dev = torch.device("cpu")
        obs_t = dict_to_tensors(obs, dev)
        m = CombatPPOAgent(d_model=64, nhead=4, dim_ff=128)
        m.eval()
        with torch.no_grad():
            sel, ex, v = m(obs_t)
        assert sel.shape == (3, MAX_HAND_LENGTH, 2)
        assert ex.shape == (3, 2)
        assert v.shape == (3, 1)
    finally:
        vec.close()


def test_minimal_ppo_forward_smoke():
    torch = pytest.importorskip("torch")
    from env.lite_combat_env import dict_to_tensors, make_vec
    from agent.minimal_model import MinimalCombatPPOAgent
    from environment import MAX_HAND_LENGTH

    vec = make_vec(_tiny_pool(), n=3, base_seed=0, backend="sync")
    try:
        obs, _ = vec.reset(seed=0)
        dev = torch.device("cpu")
        obs_t = dict_to_tensors(obs, dev)
        m = MinimalCombatPPOAgent(d_model=64, nhead=4, dim_ff=128)
        m.eval()
        with torch.no_grad():
            sel, ex, v = m(obs_t)
        assert sel.shape == (3, MAX_HAND_LENGTH, 2)
        assert ex.shape == (3, 2)
        assert v.shape == (3, 1)
    finally:
        vec.close()


def test_combat_ppo_agent_lazy_export():
    pytest.importorskip("torch")
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]
    agent_mod = importlib.import_module("agent")
    cls = getattr(agent_mod, "CombatPPOAgent")
    assert cls.__name__ == "CombatPPOAgent"


def test_minimal_ppo_agent_lazy_export():
    pytest.importorskip("torch")
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]
    agent_mod = importlib.import_module("agent")
    cls = getattr(agent_mod, "MinimalCombatPPOAgent")
    assert cls.__name__ == "MinimalCombatPPOAgent"
