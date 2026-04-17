"""Combat PPO policy for ``balatro_lite_gym`` flat observations (``adapt_lite_vector_obs``).

Ported from Duke ``combat_agent_model.py`` with suit-major card ids, lite joker/boss counts,
no seals or money, ``joker_mask`` + ``joker_editions``, and debuff-only hand flags.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from defs import (
    HAND_TYPE_COUNT,
    NUM_CARD_EDITIONS,
    NUM_CARD_ENHANCEMENTS,
    NUM_JOKERS,
    NUM_RANKS,
    NUM_SUITS,
)
from defs.bosses import BossBlind
from environment import CARD_ID_HIGH, MAX_HAND_LENGTH, MAX_JOKER_LENGTH

__all__ = [
    "CardEmbedding",
    "RunStateEmbedding",
    "HandLevelEmbedding",
    "ModifierEmbedding",
    "CombatEmbeddings",
    "PreNormBlock",
    "CombatBackbone",
    "CombatHeads",
    "CombatPPOAgent",
]

_NUM_BOSS_BLINDS = len(BossBlind.__members__)


class CardEmbedding(nn.Module):
    """Rank + suit + enhancement + edition; padded slots zeroed via ``slot_mask``."""

    def __init__(self, d_model: int):
        super().__init__()
        d = d_model
        self.rank_emb = nn.Embedding(NUM_RANKS, d)
        self.suit_emb = nn.Embedding(NUM_SUITS, d)
        self.enhancement_emb = nn.Embedding(NUM_CARD_ENHANCEMENTS, d)
        self.edition_emb = nn.Embedding(NUM_CARD_EDITIONS, d)

    def forward(
        self,
        card_ids: torch.Tensor,
        enhancements: torch.Tensor,
        editions: torch.Tensor,
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
        toks = (
            self.rank_emb(rank)
            + self.suit_emb(suit)
            + self.enhancement_emb(enhancements.long())
            + self.edition_emb(editions.long())
        )
        toks = toks * mask.unsqueeze(-1).float()
        return toks, mask


class RunStateEmbedding(nn.Module):
    """hands_remaining, discards_remaining, player_hand_size, log-scores (no money)."""

    NUM_SCALARS = 5

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(self.NUM_SCALARS, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, obs: dict) -> torch.Tensor:
        feats = torch.stack(
            [
                obs["hands_remaining"].float().squeeze(-1),
                obs["discards_remaining"].float().squeeze(-1),
                obs["player_hand_size"].float().squeeze(-1),
                torch.log1p(obs["current_score"].float().squeeze(-1)),
                torch.log1p(obs["target_score"].float().squeeze(-1)),
            ],
            dim=-1,
        )
        return self.ln(self.proj(feats).unsqueeze(1))


class HandLevelEmbedding(nn.Module):
    """Hand-type rows: embed type id + project chip/mult (columns 1–2)."""

    _HL_FEATS = 2

    def __init__(self, d_model: int):
        super().__init__()
        self.type_emb = nn.Embedding(HAND_TYPE_COUNT, d_model)
        self.level_proj = nn.Linear(self._HL_FEATS, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, hand_levels: torch.Tensor) -> torch.Tensor:
        ht_ids = hand_levels[:, :, 0].long()
        hl_feats = hand_levels[:, :, 1:].float()
        return self.ln(self.type_emb(ht_ids) + self.level_proj(hl_feats))


class ModifierEmbedding(nn.Module):
    """Boss + jokers: shared id table; joker editions added; pad slots zeroed like cards.

    Id layout: jokers ``0 .. NUM_JOKERS - 1`` (padded joker slots are id 0 from obs);
    bosses ``NUM_JOKERS .. NUM_JOKERS + NUM_BOSS - 1``. After ``LayerNorm``, tokens are
    multiplied by ``mod_mask`` so inactive slots (same trick as :class:`CardEmbedding`).
    """

    _NUM_MODIFIER_IDS = NUM_JOKERS + _NUM_BOSS_BLINDS
    _MAX_MODIFIERS = 1 + MAX_JOKER_LENGTH

    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(self._NUM_MODIFIER_IDS, d_model)
        self.pos_emb = nn.Embedding(self._MAX_MODIFIERS, d_model)
        self.edition_emb = nn.Embedding(NUM_CARD_EDITIONS, d_model, padding_idx=0)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs["boss_id"].device
        b = obs["boss_id"].shape[0]
        has_boss = obs["boss_is_active"].squeeze(-1).bool()
        boss_raw = obs["boss_id"].long().squeeze(-1)
        bad_boss = has_boss & ((boss_raw < 0) | (boss_raw > _NUM_BOSS_BLINDS - 1))
        if bad_boss.any():
            raise ValueError(
                "boss_id out of range for rows with boss_is_active "
                f"(expected 0..{_NUM_BOSS_BLINDS - 1})"
            )
        boss_slot = (boss_raw + NUM_JOKERS).unsqueeze(1)

        joker_ids = obs["joker_ids"].long()
        joker_mask = obs["joker_mask"].bool()
        if (joker_ids < 0).any() or (joker_ids > NUM_JOKERS - 1).any():
            raise ValueError(
                f"joker_ids out of [0, {NUM_JOKERS - 1}] "
                f"(min={int(joker_ids.min())}, max={int(joker_ids.max())})"
            )
        joker_toks = joker_ids
        pad = torch.zeros(b, 1, dtype=torch.long, device=device)

        mod_ids = torch.where(
            has_boss[:, None],
            torch.cat([boss_slot, joker_toks], dim=1),
            torch.cat([joker_toks, pad], dim=1),
        )

        joker_ed = obs["joker_editions"].long()
        if (joker_ed < 0).any() or (joker_ed > NUM_CARD_EDITIONS - 1).any():
            raise ValueError(
                f"joker_editions out of [0, {NUM_CARD_EDITIONS - 1}] "
                f"(min={int(joker_ed.min())}, max={int(joker_ed.max())})"
            )
        boss_ed = torch.zeros(b, 1, dtype=torch.long, device=device)
        edition_slots = torch.where(
            has_boss[:, None],
            torch.cat([boss_ed, joker_ed], dim=1),
            torch.cat([joker_ed, pad], dim=1),
        )

        joker_real = joker_mask
        mod_mask = torch.where(
            has_boss[:, None],
            torch.cat(
                [
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    joker_real,
                ],
                dim=1,
            ),
            torch.cat(
                [joker_real, torch.zeros(b, 1, dtype=torch.bool, device=device)],
                dim=1,
            ),
        )
        no_mod = ~mod_mask.any(dim=1)
        mod_mask[:, 0] = mod_mask[:, 0] | no_mod

        pos = self.pos_emb(torch.arange(self._MAX_MODIFIERS, device=device))
        mod_seq = self.ln(
            self.emb(mod_ids)
            + pos
            + self.edition_emb(edition_slots)
        )
        mod_seq = mod_seq * mod_mask.unsqueeze(-1).float()

        return mod_seq, mod_mask


class CombatEmbeddings(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d = d_model
        self.hand_card_emb = CardEmbedding(d)
        self.deck_card_emb = CardEmbedding(d)
        self.hand_flags_proj = nn.Linear(1, d, bias=False)
        self.run_emb = RunStateEmbedding(d)
        self.hl_emb = HandLevelEmbedding(d)
        self.mod_emb = ModifierEmbedding(d)
        self.hand_ln = nn.LayerNorm(d)
        self.deck_ln = nn.LayerNorm(d)

    def forward(self, obs: dict):
        hand_mask = obs["hand_card_mask"].bool()
        hand_toks, _ = self.hand_card_emb(
            obs["hand_card_ids"].long(),
            obs["hand_card_enhancements"].long(),
            obs["hand_card_editions"].long(),
            hand_mask,
        )
        deb = obs["hand_is_debuffed"].float()
        if deb.dim() != 2:
            raise ValueError(
                f"hand_is_debuffed must be (B, {MAX_HAND_LENGTH}), got {tuple(deb.shape)}"
            )
        deb = deb.unsqueeze(-1)
        hand_toks = hand_toks + self.hand_flags_proj(deb) * hand_mask.unsqueeze(-1).float()
        hand_toks = self.hand_ln(hand_toks)

        deck_mask = obs["deck_card_mask"].bool()
        deck_toks, _ = self.deck_card_emb(
            obs["deck_card_ids"].long(),
            obs["deck_card_enhancements"].long(),
            obs["deck_card_editions"].long(),
            deck_mask,
        )
        deck_toks = self.deck_ln(deck_toks)

        run_tok = self.run_emb(obs)
        hl_toks = self.hl_emb(obs["hand_levels"])
        mod_seq, mod_mask = self.mod_emb(obs)

        b = hand_toks.shape[0]
        ctx_seq = torch.cat([hl_toks, deck_toks], dim=1)
        ctx_mask = torch.cat(
            [
                torch.ones(
                    b,
                    hl_toks.shape[1],
                    dtype=torch.bool,
                    device=hand_toks.device,
                ),
                deck_mask,
            ],
            dim=1,
        )

        return hand_toks, hand_mask, run_tok, ctx_seq, ctx_mask, mod_seq, mod_mask


class PreNormBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_norm = self.norm1(q)
        if kv is None:
            k = v = q_norm
        else:
            k = v = kv
        pad_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.attn(q_norm, k, v, key_padding_mask=pad_mask)
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q


class CombatBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        depth_hand: int = 4,
        depth_mod: int = 3,
        depth_mod_run: int = 1,
        depth_hm: int = 2,
        depth_hc: int = 2,
    ):
        super().__init__()

        def blk() -> PreNormBlock:
            return PreNormBlock(d_model, nhead, dim_ff, dropout)

        self.hand_self_layers = nn.ModuleList([blk() for _ in range(depth_hand)])
        self.mod_self_layers = nn.ModuleList([blk() for _ in range(depth_mod)])
        self.mod_run_layers = nn.ModuleList([blk() for _ in range(depth_mod_run)])
        self.hand_mod_layers = nn.ModuleList([blk() for _ in range(depth_hm)])
        self.hand_ctx_layers = nn.ModuleList([blk() for _ in range(depth_hc)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hand_toks: torch.Tensor,
        hand_mask: torch.Tensor,
        run_tok: torch.Tensor,
        ctx_seq: torch.Tensor,
        ctx_mask: torch.Tensor,
        mod_seq: torch.Tensor,
        mod_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hand = hand_toks
        for layer in self.hand_self_layers:
            hand = layer(hand, kv_mask=hand_mask)

        mod = mod_seq
        for layer in self.mod_self_layers:
            mod = layer(mod, kv_mask=mod_mask)

        for layer in self.mod_run_layers:
            mod = layer(mod, kv=run_tok)

        for layer in self.hand_mod_layers:
            hand = layer(hand, kv=mod, kv_mask=mod_mask)

        for layer in self.hand_ctx_layers:
            hand = layer(hand, kv=ctx_seq, kv_mask=ctx_mask)

        hand_final = self.final_norm(hand)
        mask_f = hand_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1)
        if (denom <= 0).any():
            raise ValueError("hand_mask has no active card in at least one batch row")
        global_ctx = (hand_final * mask_f).sum(dim=1) / denom
        return hand_final, global_ctx


class CombatHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.select_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.exec_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        hand_final: torch.Tensor,
        global_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sel_logits = self.select_head(hand_final)
        exec_logits = self.exec_head(global_ctx)
        value = self.value_head(global_ctx)
        return sel_logits, exec_logits, value


class CombatPPOAgent(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = CombatEmbeddings(d_model=d_model)
        self.backbone = CombatBackbone(
            d_model=d_model, nhead=nhead, dim_ff=dim_ff, dropout=dropout
        )
        self.heads = CombatHeads(d_model=d_model)

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            hand_toks,
            hand_mask,
            run_tok,
            ctx_seq,
            ctx_mask,
            mod_seq,
            mod_mask,
        ) = self.embeddings(obs)
        hand_final, global_ctx = self.backbone(
            hand_toks,
            hand_mask,
            run_tok,
            ctx_seq,
            ctx_mask,
            mod_seq,
            mod_mask,
        )
        return self.heads(hand_final, global_ctx)
