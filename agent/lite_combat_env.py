"""Vector combat env over ``balatro_lite_gym.BalatroEnv``: flat obs adapter, ``make_vec``, rollout buffer."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from defs import CARD_EDITION_HIGH, CARD_ENHANCEMENT_HIGH, HAND_TYPE_COUNT, JOKER_ID_HIGH
from defs.bosses import BossBlind
from engine import GameSnapshot
from environment import (
    CARD_ID_HIGH,
    MAX_DECK_LENGTH,
    MAX_HAND_LENGTH,
    MAX_JOKER_LENGTH,
    BalatroEnv,
)

# Reserved hand/deck seal channel in flat obs; lite snapshots do not populate it.
CARD_SEAL_HIGH = 0


def _batched(obs: dict, key: str) -> np.ndarray:
    return np.asarray(obs[key], dtype=np.float64)


def _nested_batched(obs: dict, *path: str) -> np.ndarray:
    cur: dict | np.ndarray = obs
    for p in path:
        cur = cur[p]
    return np.asarray(cur)


def _squeeze_scalars(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim >= 1 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def adapt_lite_vector_obs(obs: dict) -> dict[str, np.ndarray]:
    """Nested batched ``BalatroEnv`` obs → flat dict (leading batch dim ``N``)."""
    hand_ids = _nested_batched(obs, "hand", "hand_card_ids")
    hand_enh = _nested_batched(obs, "hand", "hand_enhancements")
    hand_ed = _nested_batched(obs, "hand", "hand_editions")
    hand_mask = _nested_batched(obs, "hand", "hand_mask")

    deck_ids = _nested_batched(obs, "deck", "deck_card_ids")
    deck_enh = _nested_batched(obs, "deck", "deck_enhancements")
    deck_ed = _nested_batched(obs, "deck", "deck_editions")
    deck_mask = _nested_batched(obs, "deck", "deck_mask")

    joker_ids = _nested_batched(obs, "jokers", "joker_ids")
    joker_mask = _nested_batched(obs, "jokers", "joker_mask")

    def _col(key: str) -> np.ndarray:
        x = _squeeze_scalars(_batched(obs, key)).astype(np.int64).reshape(-1)
        return x[:, np.newaxis]

    target_score = _col("target_score")
    current_score = _col("current_score")
    blind_id = _col("blind_id")
    play_rem = _col("play_remaining")
    disc_rem = _col("discard_remaining")

    hl = np.asarray(obs["hand_levels"], dtype=np.int64)
    if hl.ndim == 2:
        hl = hl[np.newaxis, ...]

    n = hand_ids.shape[0]
    hand_ids_out = hand_ids.astype(np.int64).copy()
    hand_ids_out[hand_mask == 0] = -1

    deck_ids_out = deck_ids.astype(np.int64).copy()
    deck_ids_out[deck_mask == 0] = -1

    pad_h = MAX_HAND_LENGTH - hand_ids_out.shape[1]
    if pad_h < 0:
        raise ValueError(
            f"hand length {hand_ids_out.shape[1]} > MAX_HAND_LENGTH {MAX_HAND_LENGTH}"
        )
    if pad_h > 0:
        z = np.zeros((n, pad_h), dtype=np.int64)
        m0 = np.zeros((n, pad_h), dtype=np.int64)
        hand_ids_out = np.concatenate([hand_ids_out, z - 1], axis=1)
        hand_enh = np.concatenate([hand_enh, m0], axis=1)
        hand_ed = np.concatenate([hand_ed, m0], axis=1)
        hand_mask = np.concatenate([hand_mask, m0], axis=1)

    pad_d = MAX_DECK_LENGTH - deck_ids_out.shape[1]
    if pad_d < 0:
        raise ValueError(
            f"deck length {deck_ids_out.shape[1]} > MAX_DECK_LENGTH {MAX_DECK_LENGTH}"
        )
    if pad_d > 0:
        z = np.zeros((n, pad_d), dtype=np.int64)
        deck_ids_out = np.concatenate([deck_ids_out, z - 1], axis=1)
        deck_enh = np.concatenate([deck_enh, z], axis=1)
        deck_ed = np.concatenate([deck_ed, z], axis=1)
        deck_mask = np.concatenate([deck_mask, z], axis=1)

    pad_j = MAX_JOKER_LENGTH - joker_ids.shape[1]
    if pad_j < 0:
        raise ValueError("too many jokers for flat layout")
    if pad_j > 0:
        z = np.zeros((n, pad_j), dtype=np.int64)
        joker_ids = np.concatenate([joker_ids, z], axis=1)
        joker_mask = np.concatenate([joker_mask, z], axis=1)

    ht_ids = np.broadcast_to(
        np.arange(HAND_TYPE_COUNT, dtype=np.int64), (n, HAND_TYPE_COUNT)
    )
    hl_model = np.stack(
        [
            ht_ids,
            np.zeros((n, HAND_TYPE_COUNT), dtype=np.int64),
            hl[:, :, 0],
            hl[:, :, 1],
        ],
        axis=-1,
    )

    boss_active = (blind_id >= 0).astype(np.int64)
    boss_slot_id = np.where(blind_id < 0, 0, blind_id).astype(np.int64)

    money = np.zeros((n, 1), dtype=np.int64)

    return {
        "hand_card_ids": hand_ids_out,
        "hand_card_enhancements": hand_enh.astype(np.int64),
        "hand_card_editions": hand_ed.astype(np.int64),
        "hand_card_seals": np.zeros((n, MAX_HAND_LENGTH), dtype=np.int64),
        "hand_is_face_down": np.zeros((n, MAX_HAND_LENGTH), dtype=np.uint8),
        "hand_is_debuffed": np.zeros((n, MAX_HAND_LENGTH), dtype=np.uint8),
        "deck_card_ids": deck_ids_out,
        "deck_card_enhancements": deck_enh.astype(np.int64),
        "deck_card_editions": deck_ed.astype(np.int64),
        "deck_card_seals": np.zeros((n, MAX_DECK_LENGTH), dtype=np.int64),
        "hands_remaining": play_rem,
        "discards_remaining": disc_rem,
        "money": money,
        "current_score": current_score,
        "target_score": target_score,
        "hand_levels": hl_model,
        "boss_id": boss_slot_id,
        "boss_is_active": boss_active,
        "joker_ids": joker_ids.astype(np.int64),
        "joker_is_empty": (1 - joker_mask.astype(np.uint8)).astype(np.uint8),
    }


def build_training_observation_space() -> spaces.Dict:
    """Gym space for :func:`adapt_lite_vector_obs` (single env shapes)."""
    _i64 = np.iinfo(np.int64)
    return spaces.Dict(
        {
            "hand_card_ids": spaces.Box(
                -1, CARD_ID_HIGH, (MAX_HAND_LENGTH,), dtype=np.int64
            ),
            "hand_card_enhancements": spaces.Box(
                0, CARD_ENHANCEMENT_HIGH, (MAX_HAND_LENGTH,), dtype=np.int64
            ),
            "hand_card_editions": spaces.Box(
                0, CARD_EDITION_HIGH, (MAX_HAND_LENGTH,), dtype=np.int64
            ),
            "hand_card_seals": spaces.Box(
                0, CARD_SEAL_HIGH, (MAX_HAND_LENGTH,), dtype=np.int64
            ),
            "hand_is_face_down": spaces.MultiBinary(MAX_HAND_LENGTH),
            "hand_is_debuffed": spaces.MultiBinary(MAX_HAND_LENGTH),
            "deck_card_ids": spaces.Box(
                -1, CARD_ID_HIGH, (MAX_DECK_LENGTH,), dtype=np.int64
            ),
            "deck_card_enhancements": spaces.Box(
                0, CARD_ENHANCEMENT_HIGH, (MAX_DECK_LENGTH,), dtype=np.int64
            ),
            "deck_card_editions": spaces.Box(
                0, CARD_EDITION_HIGH, (MAX_DECK_LENGTH,), dtype=np.int64
            ),
            "deck_card_seals": spaces.Box(
                0, CARD_SEAL_HIGH, (MAX_DECK_LENGTH,), dtype=np.int64
            ),
            "hands_remaining": spaces.Box(0, _i64.max, (1,), dtype=np.int64),
            "discards_remaining": spaces.Box(0, _i64.max, (1,), dtype=np.int64),
            "money": spaces.Box(_i64.min, _i64.max, (1,), dtype=np.int64),
            "current_score": spaces.Box(0, _i64.max, (1,), dtype=np.int64),
            "target_score": spaces.Box(0, _i64.max, (1,), dtype=np.int64),
            "hand_levels": spaces.Box(
                0, _i64.max, (HAND_TYPE_COUNT, 4), dtype=np.int64
            ),
            "boss_id": spaces.Box(0, int(BossBlind.THE_PLANT), (1,), dtype=np.int64),
            "boss_is_active": spaces.Box(0, 1, (1,), dtype=np.int64),
            "joker_ids": spaces.Box(
                0, JOKER_ID_HIGH, (MAX_JOKER_LENGTH,), dtype=np.int64
            ),
            "joker_is_empty": spaces.MultiBinary(MAX_JOKER_LENGTH),
        }
    )


def _strip_unit_batch(flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove leading ``N=1`` batch dim so a single env matches ``observation_space``."""
    out: dict[str, np.ndarray] = {}
    for k, v in flat.items():
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == 1:
            out[k] = a[0]
        else:
            out[k] = a
    return out


class LitePooledCombatEnv(gym.Env):
    """Sample a ``GameSnapshot`` from a pool each ``reset``; expose adapted flat observations.

    Actions are ``MultiBinary(MAX_HAND_LENGTH + 1)``: first ``MAX_HAND_LENGTH`` bits are card
    selection; last bit is execution with **0 = play**, **1 = discard** (same convention
    as the previous ``CombatActionWrapper`` / ``cs590_env`` stack).

    This matches ``balatro_lite_gym`` semantics: ``action_type`` 1 = play, 0 = discard, so
    the last action bit is mapped as ``play = 1 - last_bit`` internally.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        snapshot_pool: Sequence[GameSnapshot],
        pool_seed: int = 0,
    ) -> None:
        super().__init__()
        if not snapshot_pool:
            raise ValueError("snapshot_pool must be non-empty")
        self._pool = [deepcopy(s) for s in snapshot_pool]
        self._rng = np.random.default_rng(pool_seed)

        self.observation_space = build_training_observation_space()
        self.action_space = spaces.MultiBinary(MAX_HAND_LENGTH + 1)

        self._env = BalatroEnv(deepcopy(self._pool[0]))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        idx = int(self._rng.integers(0, len(self._pool)))
        snap = deepcopy(self._pool[idx])
        fresh_seed = int(self._rng.integers(0, 2**31))
        obs_raw, _ = self._env.reset(seed=fresh_seed, options={"snapshot": snap})
        obs = _strip_unit_batch(adapt_lite_vector_obs(self._wrap_single_obs(obs_raw)))
        return obs, {"snapshot_idx": idx, "env_seed": fresh_seed}

    @staticmethod
    def _wrap_single_obs(obs_raw: dict) -> dict:
        """Add batch dim N=1 for :func:`adapt_lite_vector_obs`."""
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

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        card_sel = action[:MAX_HAND_LENGTH].astype(np.int8)
        execution = int(action[MAX_HAND_LENGTH])
        # Old stack: 0 = play, 1 = discard. Lite: 1 = play, 0 = discard.
        action_type = 1 if execution == 0 else 0
        act = {"selection": card_sel, "action_type": int(action_type)}

        obs_raw, reward, terminated, truncated, info = self._env.step(act)
        done = bool(terminated or truncated)

        obs_adapted = _strip_unit_batch(adapt_lite_vector_obs(self._wrap_single_obs(obs_raw)))
        if done:
            obs_adapted, _ = self.reset()

        return obs_adapted, float(reward), done, False, info

    def close(self) -> None:
        return None


def make_lite_pooled_combat_env(
    snapshot_pool: Sequence[GameSnapshot],
    pool_seed: int = 0,
) -> LitePooledCombatEnv:
    """Factory for ``gymnasium.vector.VectorEnv``."""
    return LitePooledCombatEnv(snapshot_pool, pool_seed=pool_seed)


def make_vec_sync(snapshot_pool: Sequence[GameSnapshot], n: int, base_seed: int = 0):
    """Build a ``SyncVectorEnv`` with ``n`` workers (single process, sequential stepping)."""
    return _make_vec_fns(snapshot_pool, n, base_seed, sync=True)


def make_vec_async(snapshot_pool: Sequence[GameSnapshot], n: int, base_seed: int = 0):
    """Build an ``AsyncVectorEnv`` with ``n`` subprocess workers (parallel stepping)."""
    return _make_vec_fns(snapshot_pool, n, base_seed, sync=False)


def _make_vec_fns(
    snapshot_pool: Sequence[GameSnapshot],
    n: int,
    base_seed: int,
    *,
    sync: bool,
) -> gym.vector.VectorEnv:
    fns = [
        partial(make_lite_pooled_combat_env, snapshot_pool, pool_seed=base_seed + i)
        for i in range(n)
    ]
    if sync:
        return gym.vector.SyncVectorEnv(fns)
    return gym.vector.AsyncVectorEnv(fns)


def make_vec(
    snapshot_pool: Sequence[GameSnapshot],
    n: int,
    base_seed: int = 0,
    *,
    backend: str = "async",
) -> gym.vector.VectorEnv:
    """Build a vector env over ``snapshot_pool`` (one ``LitePooledCombatEnv`` per worker).

    Args:
        snapshot_pool: Picklable sequence of ``GameSnapshot`` (each worker gets a copy via
            the factory; pass a tuple or list built from plain dataclass snapshots).
        n: Number of parallel environments.
        base_seed: Worker ``i`` uses ``pool_seed=base_seed+i`` for pool draw / ordering.
        backend: ``\"async\"`` (default) for ``AsyncVectorEnv``, or ``\"sync\"`` for
            ``SyncVectorEnv`` (debug / no multiprocessing).

    Returns:
        A ``gymnasium.vector.VectorEnv`` instance.
    """
    b = backend.lower().strip()
    if b not in ("async", "sync"):
        raise ValueError(f"backend must be 'async' or 'sync', got {backend!r}")
    return _make_vec_fns(snapshot_pool, n, base_seed, sync=(b == "sync"))


# -----------------------------------------------------------------------------
# Vector rollout buffer + GAE (PyTorch). Imports ``torch`` only when these run.
# -----------------------------------------------------------------------------


def dict_to_tensors(obs_np: dict, dev):  # dev: torch.device
    import torch

    return {k: torch.as_tensor(np.asarray(v), device=dev) for k, v in obs_np.items()}


class VecRolloutBuffer:
    """One parallel rollout chunk (``T`` × ``N``): obs, actions, log-probs, values, rewards."""

    def __init__(self, T: int, N: int, dev):
        import torch

        self.T, self.N, self.dev = T, N, dev
        self.card_sels = torch.zeros(T, N, MAX_HAND_LENGTH, device=dev, dtype=torch.long)
        self.executions = torch.zeros(T, N, device=dev, dtype=torch.long)
        self.log_probs = torch.zeros(T, N, device=dev)
        self.values = torch.zeros(T, N, device=dev)
        self.rewards = torch.zeros(T, N, device=dev)
        self.dones = torch.zeros(T, N, device=dev)
        self.obs: dict = {}

    def store_step(
        self,
        t: int,
        obs_t: dict,
        card_sels,
        executions,
        log_probs,
        values,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        import torch

        if not self.obs:
            self.obs = {
                k: torch.zeros(self.T, self.N, *v.shape[1:], device=self.dev, dtype=v.dtype)
                for k, v in obs_t.items()
            }
        for k, v in obs_t.items():
            self.obs[k][t] = v.detach()
        self.card_sels[t] = card_sels.detach()
        self.executions[t] = executions.detach()
        self.log_probs[t] = log_probs.detach()
        self.values[t] = values.detach()
        self.rewards[t] = torch.as_tensor(rewards, device=self.dev)
        self.dones[t] = torch.as_tensor(dones.astype(np.float32), device=self.dev)

    def flatten(self):
        TN = self.T * self.N
        flat_obs = {k: v.reshape(TN, *v.shape[2:]) for k, v in self.obs.items()}
        return (
            flat_obs,
            self.card_sels.reshape(TN, MAX_HAND_LENGTH),
            self.executions.reshape(TN),
            self.log_probs.reshape(TN),
        )


def compute_gae_vectorized(
    rewards,
    values,
    next_values,
    dones,
    gamma: float,
    gae_lambda: float,
):
    import torch

    with torch.no_grad():
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device=rewards.device)

        for t in reversed(range(T)):
            next_val = next_values if t == T - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * not_done - values[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
