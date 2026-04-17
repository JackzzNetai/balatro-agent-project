"""Minimal combat PPO policy: hand cards (rank+suit only), play/discard remaining, hand levels.

Uses the same flat observation dict as :func:`env.lite_combat_env.adapt_lite_vector_obs` but
ignores deck, jokers, boss, scores, and card modifiers. Forward API matches :class:`CombatPPOAgent`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from defs import HAND_TYPE_COUNT, NUM_RANKS, NUM_SUITS
from environment import CARD_ID_HIGH

from .model import CombatHeads, HandLevelEmbedding, PreNormBlock

__all__ = [
    "SimpleCardEmbedding",
    "MinimalRunStateEmbedding",
    "MinimalCombatEmbeddings",
    "MinimalCombatBackbone",
    "MinimalCombatPPOAgent",
]


class SimpleCardEmbedding(nn.Module):
    """Rank + suit only; padded slots zeroed via ``slot_mask``."""

    def __init__(self, d_model: int):
        super().__init__()
        d = d_model
        self.rank_emb = nn.Embedding(NUM_RANKS, d)
        self.suit_emb = nn.Embedding(NUM_SUITS, d)

    def forward(
        self,
        card_ids: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = slot_mask.bool()
        if (card_ids < 0).any() or (card_ids > CARD_ID_HIGH).any():
            raise ValueError(
                f"card_ids must be in [0, {CARD_ID_HIGH}], got min/max "
                f"{int(card_ids.min())}/{int(card_ids.max())}"
            )
        rank = (card_ids % NUM_RANKS).long()
        suit = (card_ids // NUM_RANKS).long()
        toks = self.rank_emb(rank) + self.suit_emb(suit)
        toks = toks * mask.unsqueeze(-1).float()
        return toks, mask


class MinimalRunStateEmbedding(nn.Module):
    """``hands_remaining`` and ``discards_remaining`` → one ``(B, 1, d_model)`` token."""

    NUM_SCALARS = 2

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(self.NUM_SCALARS, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, obs: dict) -> torch.Tensor:
        feats = torch.stack(
            [
                obs["hands_remaining"].float().squeeze(-1),
                obs["discards_remaining"].float().squeeze(-1),
            ],
            dim=-1,
        )
        return self.ln(self.proj(feats).unsqueeze(1))


class MinimalCombatEmbeddings(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.hand_card_emb = SimpleCardEmbedding(d_model)
        self.run_emb = MinimalRunStateEmbedding(d_model)
        self.hl_emb = HandLevelEmbedding(d_model)
        self.hand_ln = nn.LayerNorm(d_model)

    def forward(self, obs: dict):
        hand_mask = obs["hand_card_mask"].bool()
        hand_toks, _ = self.hand_card_emb(
            obs["hand_card_ids"].long(),
            hand_mask,
        )
        hand_toks = self.hand_ln(hand_toks)

        run_tok = self.run_emb(obs)
        hl_toks = self.hl_emb(obs["hand_levels"])

        ctx_seq = torch.cat([hl_toks, run_tok], dim=1)
        b = hand_toks.shape[0]
        ctx_mask = torch.ones(
            b,
            HAND_TYPE_COUNT + 1,
            dtype=torch.bool,
            device=hand_toks.device,
        )
        return hand_toks, hand_mask, ctx_seq, ctx_mask


class MinimalCombatBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        depth_hand: int = 3,
        depth_hc: int = 2,
    ):
        super().__init__()

        def blk() -> PreNormBlock:
            return PreNormBlock(d_model, nhead, dim_ff, dropout)

        self.hand_self_layers = nn.ModuleList([blk() for _ in range(depth_hand)])
        self.hand_ctx_layers = nn.ModuleList([blk() for _ in range(depth_hc)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hand_toks: torch.Tensor,
        hand_mask: torch.Tensor,
        ctx_seq: torch.Tensor,
        ctx_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hand = hand_toks
        for layer in self.hand_self_layers:
            hand = layer(hand, kv_mask=hand_mask)

        for layer in self.hand_ctx_layers:
            hand = layer(hand, kv=ctx_seq, kv_mask=ctx_mask)

        hand_final = self.final_norm(hand)
        mask_f = hand_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1)
        if (denom <= 0).any():
            raise ValueError("hand_mask has no active card in at least one batch row")
        global_ctx = (hand_final * mask_f).sum(dim=1) / denom
        return hand_final, global_ctx


class MinimalCombatPPOAgent(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        *,
        depth_hand: int = 2,
        depth_hc: int = 2,
    ):
        super().__init__()
        self.embeddings = MinimalCombatEmbeddings(d_model=d_model)
        self.backbone = MinimalCombatBackbone(
            d_model=d_model,
            nhead=nhead,
            dim_ff=dim_ff,
            dropout=dropout,
            depth_hand=depth_hand,
            depth_hc=depth_hc,
        )
        self.heads = CombatHeads(d_model=d_model)

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_toks, hand_mask, ctx_seq, ctx_mask = self.embeddings(obs)
        hand_final, global_ctx = self.backbone(
            hand_toks,
            hand_mask,
            ctx_seq,
            ctx_mask,
        )
        return self.heads(hand_final, global_ctx)
