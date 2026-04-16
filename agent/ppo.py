"""PPO helpers: action masking, log-probs / entropy, clipped surrogate update."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import MAX_HAND_LENGTH

from .model import CombatPPOAgent
from .ppo_config import PPOConfig

__all__ = [
    "get_card_mask",
    "mask_logits",
    "compute_log_prob_and_entropy",
    "ppo_update",
]


def get_card_mask(obs_t: dict[str, Any]) -> torch.Tensor:
    """``(B, MAX_HAND_LENGTH)`` bool — True for real cards, False for padding."""
    if "hand_card_mask" not in obs_t:
        raise KeyError("obs_t must include 'hand_card_mask' for lite flat obs")
    m = obs_t["hand_card_mask"].bool()
    if m.ndim != 2 or m.shape[1] != MAX_HAND_LENGTH:
        raise ValueError(
            f"hand_card_mask must be (B, {MAX_HAND_LENGTH}), got {tuple(m.shape)}"
        )
    return m


def mask_logits(sel_logits: torch.Tensor, card_mask: torch.Tensor) -> torch.Tensor:
    """Mask the **select** logit (class 1) for empty hand slots (Duke convention)."""
    if sel_logits.ndim != 3 or sel_logits.shape[-1] != 2:
        raise ValueError(
            f"sel_logits must be (B, H, 2), got {tuple(sel_logits.shape)}"
        )
    if card_mask.shape != sel_logits.shape[:2]:
        raise ValueError(
            f"card_mask {tuple(card_mask.shape)} != sel_logits[:2] {tuple(sel_logits.shape[:2])}"
        )
    masked = sel_logits.clone()
    masked[:, :, 1] = masked[:, :, 1].masked_fill(~card_mask, -1e9)
    return masked


def compute_log_prob_and_entropy(
    agent: CombatPPOAgent,
    obs_batch: dict[str, torch.Tensor],
    card_sels: torch.Tensor,
    executions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward policy and compute log-probs / entropy for stored actions.

    Returns:
        log_probs: (B,)
        entropy: (B,)
        values: (B,)
    """
    sel_logits, exec_logits, values = agent(obs_batch)
    card_mask = get_card_mask(obs_batch)
    sel_logits = mask_logits(sel_logits, card_mask)

    sel_dist = Categorical(logits=sel_logits)
    exec_dist = Categorical(logits=exec_logits)

    log_probs = sel_dist.log_prob(card_sels).sum(dim=-1) + exec_dist.log_prob(
        executions
    )
    entropy = sel_dist.entropy().sum(dim=-1) + exec_dist.entropy()
    return log_probs, entropy, values.squeeze(-1)


def ppo_update(
    agent: CombatPPOAgent,
    optimizer: torch.optim.Optimizer,
    obs_batch: dict[str, torch.Tensor],
    card_sels: torch.Tensor,
    executions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
    """PPO clipped surrogate over flattened rollout batch (B = T * N)."""
    b = int(advantages.shape[0])
    if b == 0:
        raise ValueError("empty advantage batch")
    if b % cfg.num_minibatches != 0:
        raise ValueError(
            f"batch size {b} must be divisible by num_minibatches={cfg.num_minibatches}"
        )
    mb_size = b // cfg.num_minibatches

    total_pg = 0.0
    total_vf = 0.0
    total_ent = 0.0
    num_updates = 0

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(b, device=advantages.device)
        for k in range(cfg.num_minibatches):
            start = k * mb_size
            mb = perm[start : start + mb_size]

            mb_obs = {key: val[mb] for key, val in obs_batch.items()}
            mb_card_sels = card_sels[mb]
            mb_executions = executions[mb]
            mb_old_lp = old_log_probs[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            curr_lp, entropy, curr_values = compute_log_prob_and_entropy(
                agent, mb_obs, mb_card_sels, mb_executions
            )

            ratio = torch.exp(curr_lp - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = (
                torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            )
            pg_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(curr_values, mb_ret)
            ent_bonus = entropy.mean()

            loss = pg_loss + cfg.c_value * value_loss - cfg.c_entropy * ent_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_pg += float(pg_loss.item())
            total_vf += float(value_loss.item())
            total_ent += float(ent_bonus.item())
            num_updates += 1

    if num_updates == 0:
        raise RuntimeError("ppo_update ran zero optimizer steps")

    n = float(num_updates)
    return {
        "pg_loss": total_pg / n,
        "value_loss": total_vf / n,
        "entropy": total_ent / n,
    }
