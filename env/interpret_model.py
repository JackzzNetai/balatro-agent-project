"""Drive one :class:`~environment.BalatroEnv` built from a snapshot JSON.

For each timestep: read the env observation, run the checkpoint policy to get an action,
call :meth:`~environment.BalatroEnv.step` with that action, print what happened. No
vector env, no replay buffer—just a single env instance and the weights you pass in.

Run from repo root::

    python env/interpret_model.py path/to/combat_ppo_iter_100.pt path/to/snapshot.json

Options::

    python env/interpret_model.py ckpt.pt snap.json --seed 0 --stochastic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "balatro_lite_gym"))

from agent.model import CombatPPOAgent  # noqa: E402
from agent.ppo import get_card_mask, mask_logits  # noqa: E402
from agent.ppo_config import PPOConfig  # noqa: E402
from env.lite_combat_env import adapt_lite_vector_obs, dict_to_tensors  # noqa: E402
from env.snapshot_io import load_snapshot  # noqa: E402
from environment import MAX_HAND_LENGTH, BalatroEnv  # noqa: E402
from defs import CARD_RANK_LABELS, CardRank  # noqa: E402
from util import rank_from_card_id, suit_from_card_id  # noqa: E402

_SUIT_GLYPHS = "♠♥♦♣"


def _nested_obs_add_batch_dim(obs_raw: dict) -> dict:
    """Match training: leading batch ``N=1`` for :func:`adapt_lite_vector_obs`."""
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


def _card_face(card_id: int) -> str:
    r = CardRank(rank_from_card_id(card_id))
    s = int(suit_from_card_id(card_id))
    return f"{CARD_RANK_LABELS[r]}{_SUIT_GLYPHS[s]}"


def _print_state(title: str, snap) -> None:
    hand = snap.hand
    faces = [_card_face(c.card_id) for c in hand]
    line = " ".join(f"{faces[i]:>3}" for i in range(len(faces)))
    print(f"\n{title}")
    print(
        f"  score {snap.current_score}/{snap.target_score}  "
        f"hands {snap.play_remaining}  discards {snap.discard_remaining}  "
        f"hand_size {len(hand)}"
    )
    print(f"  hand: {line}")
    print(f"  slots: {'  '.join(str(i) for i in range(len(hand)))}")


def _action_from_checkpoint(
    agent: CombatPPOAgent,
    obs_t: dict,
    *,
    stochastic: bool,
) -> tuple[np.ndarray, float]:
    """Policy forward (saved weights) → flat action ``(MAX_HAND_LENGTH + 1,)`` int8 + value."""
    with torch.no_grad():
        sel_logits, exec_logits, value = agent(obs_t)
    card_mask = get_card_mask(obs_t)
    sel_logits = mask_logits(sel_logits, card_mask)

    if stochastic:
        sel_dist = Categorical(logits=sel_logits)
        exec_dist = Categorical(logits=exec_logits)
        card_sels = sel_dist.sample()
        executions = exec_dist.sample()
    else:
        card_sels = sel_logits.argmax(dim=-1)
        executions = exec_logits.argmax(dim=-1)

    exec_bit = int(executions[0].item())
    action = np.concatenate(
        [card_sels[0].cpu().numpy().astype(np.int8), np.array([exec_bit], dtype=np.int8)],
        axis=0,
    )
    v = float(value[0].squeeze(-1).item())
    return action, v


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Single BalatroEnv from a snapshot JSON; each step uses the action from the checkpoint."
        )
    )
    p.add_argument(
        "checkpoint",
        type=Path,
        help="``.pt`` file from training (model + optional PPOConfig).",
    )
    p.add_argument(
        "snapshot",
        type=Path,
        help="JSON snapshot for ``GameSnapshot``.",
    )
    p.add_argument("--seed", type=int, default=0, help="Env RNG seed (draw order).")
    p.add_argument(
        "--device",
        default=None,
        help="``cuda``, ``cpu``, or omit for auto.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions; default is greedy (argmax).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Safety cap on env steps (invalid actions do not advance episode).",
    )
    args = p.parse_args()
    ckpt_path = args.checkpoint.resolve()
    snap_path = args.snapshot.resolve()
    if not ckpt_path.is_file():
        sys.exit(f"checkpoint not found: {ckpt_path}")
    if not snap_path.is_file():
        sys.exit(f"snapshot JSON not found: {snap_path}")

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    snapshot = load_snapshot(snap_path)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("config")
    if cfg is None:
        cfg = PPOConfig()
        print("warning: checkpoint has no 'config'; using default PPOConfig()", file=sys.stderr)

    agent = CombatPPOAgent(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    ).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    env = BalatroEnv(snapshot)
    obs_raw, info = env.reset(
        seed=args.seed,
        options={"snapshot": snapshot},
    )
    step = 0
    print(f"checkpoint: {ckpt_path}")
    print(f"snapshot:   {snap_path}")
    print(f"device:     {device}  |  action from policy: {'sample' if args.stochastic else 'argmax'}")

    while step < args.max_steps:
        snap = info["snapshot"]
        _print_state(f"=== Step {step} (before action) ===", snap)

        obs_t = dict_to_tensors(
            adapt_lite_vector_obs(_nested_obs_add_batch_dim(obs_raw)),
            device,
        )
        action, val = _action_from_checkpoint(agent, obs_t, stochastic=args.stochastic)

        indices = [i for i in range(MAX_HAND_LENGTH) if int(action[i]) == 1]
        n_real = len(snap.hand)
        indices = [i for i in indices if i < n_real]
        exec_bit = int(action[MAX_HAND_LENGTH])
        verb = "PLAY" if exec_bit == 0 else "DISCARD"
        picked = [_card_face(snap.hand[i].card_id) for i in indices if i < len(snap.hand)]
        print(
            f"  policy value: {val:+.4f}  |  {verb}  |  selected slots: {indices}  ->  {picked}"
        )

        card_sel = action[:MAX_HAND_LENGTH].astype(np.int8)
        execution = int(action[MAX_HAND_LENGTH])
        action_type = 1 if execution == 0 else 0
        act = {"selection": card_sel, "action_type": int(action_type)}

        obs_raw, reward, terminated, truncated, info = env.step(act)
        step += 1

        print(f"  env reward: {reward:+.4f}  |  terminated={terminated}")

        if reward < 0 and not terminated:
            print("  (invalid selection or illegal discard; env returned -1 without ending)")

        if terminated or truncated:
            snap = info["snapshot"]
            won = info.get("combat_won")
            _print_state(f"=== Terminal (step {step}) ===", snap)
            if won is not None:
                print(f"  combat_won: {won}")
            break
    else:
        print(f"\nStopped after --max-steps={args.max_steps} without terminal.")


if __name__ == "__main__":
    main()
