from __future__ import annotations

import copy
import math

import numpy as np
from gymnasium import Env, spaces

from defs import (
    CARD_EDITION_HIGH,
    CARD_ENHANCEMENT_HIGH,
    HAND_TYPE_COUNT,
    JOKER_EDITION_HIGH,
    JOKER_ID_HIGH,
    NUM_RANKS,
    NUM_SUITS,
)
from engine import Card, GameSnapshot, Joker
from scoring import score_play

# -----------------------------------------------------------------------------
# Constants — observation caps (independent of transient game state)
# -----------------------------------------------------------------------------
MAX_HAND_LENGTH = 20
MAX_DECK_LENGTH = 100
MAX_JOKER_LENGTH = 10

# Padding convention: invalid slots are zeros; `*_mask` is 1 for real entries, 0 for pad.
# `*_size` is the true count of entries (capped at the corresponding MAX_*_LENGTH for the vector).

# -----------------------------------------------------------------------------
# Constants — categorical bounds for Box spaces (inclusive for integer dtypes)
# -----------------------------------------------------------------------------
# ``card_id = suit * NUM_RANKS + rank`` → valid ids ``0 .. NUM_SUITS * NUM_RANKS - 1``.
CARD_ID_HIGH = NUM_SUITS * NUM_RANKS - 1


# -----------------------------------------------------------------------------
# RNG
# -----------------------------------------------------------------------------


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return int(np.random.default_rng().integers(0, 2**31, dtype=np.int64))
    return int(seed)


# -----------------------------------------------------------------------------
# Step / gameplay helpers
# -----------------------------------------------------------------------------


def _selected_indices(selection, hand_len: int) -> list[int]:
    out: list[int] = []
    for i in range(hand_len):
        if selection[i] == 1:
            out.append(i)
    return out


def _is_invalid_selection(indices: list[int]) -> bool:
    return len(indices) == 0 or len(indices) > 5


def _remove_selected_from_hand(hand: list[Card], indices: list[int]) -> None:
    for idx in sorted(indices, reverse=True):
        del hand[idx]


def _draw_until_hand_size(
    hand: list[Card],
    deck: list[Card],
    player_hand_size: int,
    rng: np.random.Generator,
) -> None:
    while len(hand) < player_hand_size and len(deck) > 0:
        j = rng.integers(len(deck))
        hand.append(deck.pop(j))


def _terminal_reward(play_remaining: int, current_score: int) -> float:
    """Episode-end reward: play_remaining + sqrt(log10(current_score))."""
    return play_remaining + math.sqrt(math.log10(current_score))


# -----------------------------------------------------------------------------
# Observation encoding (GameSnapshot -> dict of ndarrays)
# -----------------------------------------------------------------------------


def _scalar_int(x: int) -> np.ndarray:
    return np.array([x], dtype=np.int32)


def _encode_card_pile(
    cards: list[Card], max_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(cards) > max_len:
        raise ValueError(
            f"card pile length {len(cards)} exceeds observation cap {max_len}"
        )
    n = len(cards)
    ids = np.zeros(max_len, dtype=np.int32)
    enhancements = np.zeros(max_len, dtype=np.int32)
    editions = np.zeros(max_len, dtype=np.int32)
    mask = np.zeros(max_len, dtype=np.int32)
    for i in range(n):
        c = cards[i]
        ids[i] = c.card_id
        enhancements[i] = c.enhancement
        editions[i] = c.edition
        mask[i] = 1
    return _scalar_int(n), ids, enhancements, editions, mask


def _encode_jokers(
    jokers: list[Joker], max_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(jokers) > max_len:
        raise ValueError(
            f"joker list length {len(jokers)} exceeds observation cap {max_len}"
        )
    n = len(jokers)
    ids = np.zeros(max_len, dtype=np.int32)
    editions = np.zeros(max_len, dtype=np.int32)
    mask = np.zeros(max_len, dtype=np.int32)
    for i in range(n):
        j = jokers[i]
        ids[i] = j.id
        editions[i] = j.edition
        mask[i] = 1
    return _scalar_int(n), ids, editions, mask


def _encode_hand_levels(hand_levels: dict[int, list[int]]) -> np.ndarray:
    out = np.zeros((HAND_TYPE_COUNT, 2), dtype=np.int32)
    for hand_type_id, pair in hand_levels.items():
        if not (0 <= hand_type_id < HAND_TYPE_COUNT):
            raise ValueError(
                f"hand_levels key out of range: {hand_type_id} "
                f"(valid 0..{HAND_TYPE_COUNT - 1})"
            )
        if len(pair) != 2:
            raise ValueError(
                f"hand_levels[{hand_type_id}] must have length 2, got {len(pair)}"
            )
        out[hand_type_id, 0] = pair[0]
        out[hand_type_id, 1] = pair[1]
    return out


def snapshot_to_obs_dict(snapshot: GameSnapshot) -> dict:
    h_size, h_ids, h_enh, h_ed, h_mask = _encode_card_pile(snapshot.hand, MAX_HAND_LENGTH)
    d_size, d_ids, d_enh, d_ed, d_mask = _encode_card_pile(snapshot.deck, MAX_DECK_LENGTH)
    j_size, j_ids, j_ed, j_mask = _encode_jokers(snapshot.jokers, MAX_JOKER_LENGTH)
    return {
        "target_score": _scalar_int(snapshot.target_score),
        "current_score": _scalar_int(snapshot.current_score),
        "blind_id": _scalar_int(snapshot.blind_id),
        "hand": {
            "hand_size": h_size,
            "hand_card_ids": h_ids,
            "hand_enhancements": h_enh,
            "hand_editions": h_ed,
            "hand_mask": h_mask,
        },
        "deck": {
            "deck_size": d_size,
            "deck_card_ids": d_ids,
            "deck_enhancements": d_enh,
            "deck_editions": d_ed,
            "deck_mask": d_mask,
        },
        "jokers": {
            "joker_size": j_size,
            "joker_ids": j_ids,
            "joker_editions": j_ed,
            "joker_mask": j_mask,
        },
        "play_remaining": _scalar_int(snapshot.play_remaining),
        "discard_remaining": _scalar_int(snapshot.discard_remaining),
        "player_hand_size": _scalar_int(snapshot.player_hand_size),
        "hand_levels": _encode_hand_levels(snapshot.hand_levels),
    }


# -----------------------------------------------------------------------------
# Gymnasium spaces
# -----------------------------------------------------------------------------


def _hand_space(max_len: int, prefix: str) -> spaces.Dict:
    return spaces.Dict(
        {
            f"{prefix}_size": spaces.Box(
                low=0, high=max_len, shape=(1,), dtype=np.int32
            ),
            f"{prefix}_card_ids": spaces.Box(
                low=0, high=CARD_ID_HIGH, shape=(max_len,), dtype=np.int32
            ),
            f"{prefix}_enhancements": spaces.Box(
                low=0, high=CARD_ENHANCEMENT_HIGH, shape=(max_len,), dtype=np.int32
            ),
            f"{prefix}_editions": spaces.Box(
                low=0, high=CARD_EDITION_HIGH, shape=(max_len,), dtype=np.int32
            ),
            f"{prefix}_mask": spaces.Box(
                low=0, high=1, shape=(max_len,), dtype=np.int32
            ),
        }
    )


def _joker_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "joker_size": spaces.Box(
                low=0, high=MAX_JOKER_LENGTH, shape=(1,), dtype=np.int32
            ),
            "joker_ids": spaces.Box(
                low=0, high=JOKER_ID_HIGH, shape=(MAX_JOKER_LENGTH,), dtype=np.int32
            ),
            "joker_editions": spaces.Box(
                low=0, high=JOKER_EDITION_HIGH, shape=(MAX_JOKER_LENGTH,), dtype=np.int32
            ),
            "joker_mask": spaces.Box(
                low=0, high=1, shape=(MAX_JOKER_LENGTH,), dtype=np.int32
            ),
        }
    )


def build_observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "target_score": spaces.Box(
                low=np.iinfo(np.int32).min,
                high=np.iinfo(np.int32).max,
                shape=(1,),
                dtype=np.int32,
            ),
            "current_score": spaces.Box(
                low=np.iinfo(np.int32).min,
                high=np.iinfo(np.int32).max,
                shape=(1,),
                dtype=np.int32,
            ),
            "blind_id": spaces.Box(
                low=np.iinfo(np.int32).min,
                high=np.iinfo(np.int32).max,
                shape=(1,),
                dtype=np.int32,
            ),
            "hand": _hand_space(MAX_HAND_LENGTH, "hand"),
            "deck": _hand_space(MAX_DECK_LENGTH, "deck"),
            "jokers": _joker_space(),
            "play_remaining": spaces.Box(
                low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
            ),
            "discard_remaining": spaces.Box(
                low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
            ),
            "player_hand_size": spaces.Box(
                low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
            ),
            "hand_levels": spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(HAND_TYPE_COUNT, 2),
                dtype=np.int32,
            ),
        }
    )


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class BalatroEnv(Env):
    """Balatro-Lite Gymnasium environment.

    **Construction:** pass a ``GameSnapshot``. The env keeps ``_init_snapshot_template``
    as a ``deepcopy`` (frozen baseline for bare ``reset``) and aliases ``_snapshot`` to
    your object until the next ``reset`` replaces it—only this env should mutate that
    object while aliased.

    **``reset(seed, options)``:** ``seed=None`` picks a random integer seed (same rule as
    Gymnasium RNG setup). Pass ``options["snapshot"] = S`` to install a new layout:
    ``_init_snapshot_template = deepcopy(S)``, ``_snapshot = S`` (reference). Omit
    snapshot or pass ``None`` to set ``_snapshot = deepcopy(_init_snapshot_template)``
    (fresh episode from the last installed template, or the constructor baseline).
    """

    metadata = {"render_modes": []}

    def __init__(self, snapshot: GameSnapshot) -> None:
        super().__init__()
        self.observation_space = build_observation_space()
        self.action_space = spaces.Dict(
            {
                "selection": spaces.MultiBinary(MAX_HAND_LENGTH),
                "action_type": spaces.Discrete(2),
            }
        )
        self._init_snapshot_template: GameSnapshot = copy.deepcopy(snapshot)
        self._snapshot: GameSnapshot = snapshot
        super().reset(seed=_resolve_seed(None), options=None)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = options or {}
        resolved = _resolve_seed(seed)
        super().reset(seed=resolved, options=None)
        src = opts.get("snapshot", None)
        if src is not None:
            self._init_snapshot_template = copy.deepcopy(src)
            self._snapshot = src
        else:
            self._snapshot = copy.deepcopy(self._init_snapshot_template)
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        return snapshot_to_obs_dict(self._snapshot)

    def _calculate_score(self, selected_cards: list[Card]) -> int:
        return score_play(selected_cards, self._snapshot, self.np_random)

    def step(self, action):
        selection = action["selection"]
        action_type = action["action_type"]
        snap = self._snapshot
        hand = snap.hand
        indices = _selected_indices(selection, len(hand))

        if _is_invalid_selection(indices):
            return self._get_obs(), -1, False, False, {}

        if action_type == 1:
            if snap.play_remaining == 0:
                raise RuntimeError(
                    "cannot play: play_remaining is 0 (episode already ended)"
                )
        elif action_type == 0:
            if snap.discard_remaining == 0:
                return self._get_obs(), -1, False, False, {}
        else:
            raise ValueError(f"invalid action_type: {action_type!r}")

        selected_cards = [hand[i] for i in indices]
        _remove_selected_from_hand(hand, indices)

        if action_type == 1:
            snap.play_remaining -= 1
            delta = self._calculate_score(selected_cards)
            snap.current_score += delta
        else:
            snap.discard_remaining -= 1

        terminated = (
            snap.current_score >= snap.target_score or snap.play_remaining == 0
        )
        if not terminated:
            _draw_until_hand_size(
                hand, snap.deck, snap.player_hand_size, self.np_random
            )

        if terminated:
            reward = _terminal_reward(snap.play_remaining, snap.current_score)
        else:
            reward = 0.0

        return self._get_obs(), reward, terminated, False, {}
