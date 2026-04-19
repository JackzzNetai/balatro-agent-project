"""Training hyperparameters for combat PPO.

Stable import path for pickling checkpoints: ``agent.ppo_config.PPOConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Parallelism
    num_envs: int = 256
    rollout_steps: int = 32

    # PPO
    lr: float = 5e-5
    use_lr_scheduler: bool = False
    lr_min_ratio: float = 0.1

    gae_lambda: float = 0.95

    clip_eps: float = 0.2
    clip_value_function: bool = True  # Value loss uses max of squared error vs clipped V if True
    value_clip_eps: float | None = None  # Use clip_eps if None

    ppo_epochs: int = 4
    num_minibatches: int = 4
    max_iterations: int = 5000

    c_value: float = 0.5
    c_entropy: float = 0.08
    c_entropy_end: float = 0.01  # Linear entropy schedule end value when ``use_entropy_scheduler`` is True.
    use_entropy_scheduler: bool = True  # If False, training leaves ``c_entropy`` at its configured value (no linear decay).

    max_grad_norm: float = 0.5

    d_model: int = 256
    nhead: int = 8
    dim_ff: int = 1024
    dropout: float = 0.0
