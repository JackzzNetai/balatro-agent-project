"""Parallel combat rollouts: vector envs over ``balatro_lite_gym``.

``balatro_lite_gym`` must be on ``sys.path``. Example::

    from agent import make_vec
    vec = make_vec(snapshot_pool, n=8, backend="async")

PyTorch rollout buffer / GAE (optional dependency; imports torch when used)::

    from agent.lite_combat_env import VecRolloutBuffer, compute_gae_vectorized, dict_to_tensors
"""

from __future__ import annotations

from .lite_combat_env import (
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
]


def __getattr__(name: str):
    if name == "CombatPPOAgent":
        from .model import CombatPPOAgent

        return CombatPPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
