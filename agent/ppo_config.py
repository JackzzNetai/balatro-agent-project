"""Training hyperparameters for combat PPO.

Stable import path for pickling checkpoints: ``agent.ppo_config.PPOConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Parallelism
    num_envs: int = 64
    rollout_steps: int = 16
    # PPO
    lr: float = 2.5e-4
    gamma: float = 1.0
    gae_lambda: float = 0.97
    clip_eps: float = 0.2
    # If True, value loss uses max of squared error vs clipped V (see PPO paper); eps from value_clip_eps or clip_eps.
    clip_value_function: bool = True
    value_clip_eps: float | None = None  # Use clip_eps if None
    ppo_epochs: int = 4
    num_minibatches: int = 4
    max_iterations: int = 1000
    c_value: float = 0.5
    c_entropy: float = 0.05
    max_grad_norm: float = 0.5
    use_lr_scheduler: bool = True
    lr_min_ratio: float = 0.1
    d_model: int = 256
    nhead: int = 8
    dim_ff: int = 1024
    dropout: float = 0.1
