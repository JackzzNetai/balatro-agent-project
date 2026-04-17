"""One-step probe: log selection logits / P(select) / argmax vs sample (for InterpretModel debugging).

Run from repo root (defaults to **sample**, same as PPO rollouts)::
    python env/debug_slot_probs.py path/to.ckpt path/to.json [--argmax]

Colab / Jupyter: use ``env/DebugSlotProbs.ipynb`` (Drive mount + same logging).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

# region agent log
_DEBUG_LOG = Path(__file__).resolve().parent.parent / ".cursor" / "debug-992ac8.log"


def _dbg(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "992ac8",
        "hypothesisId": hypothesis_id,
        "timestamp": int(time.time() * 1000),
        "message": message,
        "data": data,
    }
    _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


# endregion

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "balatro_lite_gym"))

from agent.model import CombatPPOAgent  # noqa: E402
from agent.ppo import get_card_mask, mask_logits  # noqa: E402
from agent.ppo_config import PPOConfig  # noqa: E402
from env.game_simulator import GameSimulator  # noqa: E402
from env.lite_combat_env import adapt_lite_vector_obs, dict_to_tensors  # noqa: E402
from environment import MAX_HAND_LENGTH  # noqa: E402


def _nested_obs_add_batch_dim(obs_raw: dict) -> dict:
    out: dict = {}
    for k, v in obs_raw.items():
        if isinstance(v, dict):
            out[k] = {
                kk: np.asarray(vv, dtype=np.float32)[np.newaxis, ...]
                for kk, vv in v.items()
            }
        else:
            out[k] = np.asarray(v, dtype=np.float32)[np.newaxis, ...]
    return out


def main() -> None:
    stochastic = "--argmax" not in sys.argv
    args = [a for a in sys.argv[1:] if a != "--argmax"]
    if len(args) < 2:
        print("usage: python env/debug_slot_probs.py ckpt.pt snap.json [--argmax]")
        sys.exit(1)
    ckpt_path = Path(args[0]).resolve()
    snap_path = Path(args[1]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _dbg("H2", "run_config", {"stochastic": stochastic, "device": str(device)})

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config") or PPOConfig()
    agent = CombatPPOAgent(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    ).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    sim = GameSimulator.from_json(snap_path, seed=0)
    obs_raw = sim.obs
    obs_t = dict_to_tensors(
        adapt_lite_vector_obs(_nested_obs_add_batch_dim(obs_raw)),
        device,
    )

    with torch.no_grad():
        sel_logits, exec_logits, value = agent(obs_t)
    card_mask = get_card_mask(obs_t)
    masked = mask_logits(sel_logits, card_mask)
    slot_p = torch.softmax(masked, dim=-1)[0, :, 1].cpu().numpy()

    n_real = int(card_mask[0].sum().item())
    raw_slice = sel_logits[0, :n_real, :].cpu().numpy().tolist()
    masked_slice = masked[0, :n_real, :].cpu().numpy().tolist()
    p_slice = slot_p[:n_real].tolist()

    _dbg("H1", "logits_per_slot_first_n", {"n_real": n_real, "raw": raw_slice, "masked": masked_slice})
    _dbg("H3", "softmax_p_select_first_n", {"p": p_slice, "p_std": float(np.std(p_slice))})

    if stochastic:
        sel_dist = Categorical(logits=masked)
        card_sels_s = sel_dist.sample()
        mode = "sample"
    else:
        card_sels_s = masked.argmax(dim=-1)
        mode = "argmax"
    n_sel = int(card_sels_s[0].sum().item())
    sel_vec = card_sels_s[0].cpu().numpy().tolist()

    _dbg("H2", "discrete_selection", {"mode": mode, "n_sel": n_sel, "card_sels": sel_vec[: max(n_real, 8)]})

    # H4: embedding vs backbone hand representation diversity (real slots)
    with torch.no_grad():
        pack = agent.embeddings(obs_t)
        hand_toks = pack[0]
        hf_in = hand_toks[0, :n_real, :]
        div_in = float(hf_in.std(dim=0).mean().item())
        hand_final, _ = agent.backbone(*pack)
        hf_out = hand_final[0, :n_real, :]
        div_out = float(hf_out.std(dim=0).mean().item())
        _dbg(
            "H4",
            "hand_repr_diversity",
            {"n_real": n_real, "embed_mean_std_dim": div_in, "hand_final_mean_std_dim": div_out},
        )

    print("logged to", _DEBUG_LOG, "| mode=", mode, "| n_sel=", n_sel, "| p_std=", float(np.std(p_slice)))


if __name__ == "__main__":
    main()
