"""Parallel combat rollouts: vector envs over ``balatro_lite_gym``.

``balatro_lite_gym`` must be on ``sys.path``. Example::

    from agent import make_vec
    vec = make_vec(snapshot_pool, n=8, backend="async")

PyTorch rollout buffer / GAE (optional dependency; imports torch when used)::

    from env.lite_combat_env import VecRolloutBuffer, compute_gae_vectorized, dict_to_tensors
"""

from __future__ import annotations

from env.lite_combat_env import (
    LitePooledCombatEnv,
    make_lite_pooled_combat_env,
    make_vec,
    make_vec_async,
    make_vec_sync,
)

__all__ = [
    "LitePooledCombatEnv",
    "make_lite_pooled_combat_env",
    "make_vec",
    "make_vec_async",
    "make_vec_sync",
    "CombatPPOAgent",
    "MinimalCombatPPOAgent",
    "PPOConfig",
    "compute_log_prob_and_entropy",
    "ppo_update",
    "get_card_mask",
    "mask_logits",
]


def __getattr__(name: str):
    if name == "CombatPPOAgent":
        from .model import CombatPPOAgent

        return CombatPPOAgent
    if name == "MinimalCombatPPOAgent":
        from .minimal_model import MinimalCombatPPOAgent

        return MinimalCombatPPOAgent
    if name == "PPOConfig":
        from .ppo_config import PPOConfig

        return PPOConfig
    if name in (
        "compute_log_prob_and_entropy",
        "ppo_update",
        "get_card_mask",
        "mask_logits",
    ):
        from . import ppo as _ppo

        return getattr(_ppo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
