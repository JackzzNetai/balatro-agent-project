"""Smoke tests for PPO helpers (``agent.ppo``)."""

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


def test_ppo_update_smoke():
    torch = pytest.importorskip("torch")
    from env.lite_combat_env import dict_to_tensors, make_vec
    from agent.model import CombatPPOAgent
    from agent.ppo import compute_log_prob_and_entropy, ppo_update
    from agent.ppo_config import PPOConfig
    from environment import MAX_HAND_LENGTH

    n = 4
    vec = make_vec(_tiny_pool(), n=n, base_seed=0, backend="sync")
    try:
        obs, _ = vec.reset(seed=0)
        dev = torch.device("cpu")
        obs_t = dict_to_tensors(obs, dev)

        agent = CombatPPOAgent(d_model=64, nhead=4, dim_ff=128)
        agent.train()
        cfg = PPOConfig(
            num_minibatches=4,
            ppo_epochs=1,
            lr=1e-3,
            d_model=64,
            nhead=4,
            dim_ff=128,
        )
        opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)

        card_sels = torch.zeros(n, MAX_HAND_LENGTH, dtype=torch.long, device=dev)
        executions = torch.zeros(n, dtype=torch.long, device=dev)

        with torch.no_grad():
            old_lp, _, _ = compute_log_prob_and_entropy(
                agent, obs_t, card_sels, executions
            )

        advantages = torch.zeros(n, device=dev)
        returns = torch.zeros(n, device=dev)

        stats = ppo_update(
            agent,
            opt,
            obs_t,
            card_sels,
            executions,
            old_lp,
            advantages,
            returns,
            cfg,
        )
        assert set(stats) == {"pg_loss", "value_loss", "entropy"}
        assert all(np.isfinite(stats[k]) for k in stats)
    finally:
        vec.close()


def test_import_ppo_config_does_not_load_torch():
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]
    if "torch" in sys.modules:
        del sys.modules["torch"]
    agent_mod = importlib.import_module("agent")
    _ = agent_mod.PPOConfig
    assert "torch" not in sys.modules


def test_lazy_ppo_exports():
    pytest.importorskip("torch")
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]
    agent_mod = importlib.import_module("agent")
    fn = getattr(agent_mod, "ppo_update")
    assert callable(fn)
