from __future__ import annotations

import copy
import math

import numpy as np
from gymnasium import Env, spaces
from gymnasium.error import ResetNeeded

from defs import (
    CARD_EDITION_HIGH,
    CARD_ENHANCEMENT_HIGH,
    HAND_TYPE_COUNT,
    HandType,
    JOKER_EDITION_HIGH,
    JOKER_ID_HIGH,
    NUM_RANKS,
    NUM_SUITS,
)
from engine import Card, GameSnapshot, Joker
from scoring import score_play
from util import chips_mult_for_hand_level, hand_debuff_mask

# -----------------------------------------------------------------------------
# Constants — observation caps (independent of transient game state)
# -----------------------------------------------------------------------------
MAX_HAND_LENGTH = 20
MAX_DECK_LENGTH = 100
MAX_JOKER_LENGTH = 10

# Invalid action (snapshot unchanged): reward for that step.
INVALID_ACTION_REWARD = -0.1

# Structural potential: ``BalatroEnv._state_potential`` (suit HHI, rank HHI, straight window).
STATE_POTENTIAL_NORM = 50  # scale raw weighted sum to keep PBRS comparable to terminal reward
STATE_POTENTIAL_W_SUIT = 1.0
STATE_POTENTIAL_W_SET = 1.0
STATE_POTENTIAL_W_STRAIGHT = 1.0

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
    """Win-only terminal shaping: play_remaining + sqrt(log10(current_score))."""
    return play_remaining + math.sqrt(math.log10(current_score))


def _poker_hand_chips_times_mult(snapshot: GameSnapshot, hand: HandType) -> float:
    """Chips × mult for the poker-hand line ``hand`` at ``snapshot.hand_levels``."""
    level = snapshot.hand_levels[int(hand)]
    c, m = chips_mult_for_hand_level(hand, level)
    return float(c * m)


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


def _encode_hand_levels(hand_levels: dict[int, int]) -> np.ndarray:
    out = np.zeros((HAND_TYPE_COUNT, 2), dtype=np.int32)
    for hand_type_id, level in hand_levels.items():
        if not (0 <= hand_type_id < HAND_TYPE_COUNT):
            raise ValueError(
                f"hand_levels key out of range: {hand_type_id} "
                f"(valid 0..{HAND_TYPE_COUNT - 1})"
            )
        c, mw = chips_mult_for_hand_level(HandType(hand_type_id), level)
        out[hand_type_id, 0] = c
        out[hand_type_id, 1] = int(mw)
    return out


def snapshot_to_obs_dict(snapshot: GameSnapshot) -> dict:
    h_size, h_ids, h_enh, h_ed, h_mask = _encode_card_pile(snapshot.hand, MAX_HAND_LENGTH)
    d_size, d_ids, d_enh, d_ed, d_mask = _encode_card_pile(snapshot.deck, MAX_DECK_LENGTH)
    j_size, j_ids, j_ed, j_mask = _encode_jokers(snapshot.jokers, MAX_JOKER_LENGTH)
    debuff_slots = hand_debuff_mask(snapshot)
    n_hand = len(snapshot.hand)
    if len(debuff_slots) != n_hand:
        raise ValueError(
            f"hand_debuff_mask length {len(debuff_slots)} != len(snapshot.hand) {n_hand}"
        )
    h_debuff = np.zeros(MAX_HAND_LENGTH, dtype=np.int32)
    for i in range(n_hand):
        h_debuff[i] = int(bool(debuff_slots[i]))
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
            "hand_is_debuffed": h_debuff,
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


def _card_pile_subspaces(max_len: int, prefix: str) -> dict[str, spaces.Box]:
    """Boxes for a padded card pile (hand or deck); no debuff flags."""
    return {
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


def _card_pile_space(max_len: int, prefix: str) -> spaces.Dict:
    return spaces.Dict(_card_pile_subspaces(max_len, prefix))


def _hand_obs_space() -> spaces.Dict:
    """Hand pile plus per-slot debuff flags (boss effects apply to hand only)."""
    sub = _card_pile_subspaces(MAX_HAND_LENGTH, "hand")
    sub["hand_is_debuffed"] = spaces.Box(
        low=0, high=1, shape=(MAX_HAND_LENGTH,), dtype=np.int32
    )
    return spaces.Dict(sub)


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
            "hand": _hand_obs_space(),
            "deck": _card_pile_space(MAX_DECK_LENGTH, "deck"),
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

    **Construction:** pass a ``GameSnapshot``. Only ``_init_snapshot_template`` is set
    (a ``deepcopy``); there is **no** active episode until :meth:`reset`.

    **Gymnasium contract:** call :meth:`reset` before the first :meth:`step`. The first
    observation comes from ``reset``, not from the constructor.

    **``reset(seed, options)``:** ``seed=None`` picks a random integer seed (same rule as
    Gymnasium RNG setup). Pass ``options["snapshot"] = S`` to install a new layout:
    ``_init_snapshot_template = deepcopy(S)``, ``_snapshot = S`` (reference). Omit
    ``snapshot`` to set ``_snapshot = deepcopy(_init_snapshot_template)`` (fresh episode
    from the last installed template, or the constructor baseline).

    **``info``:** ``reset`` and ``step`` return ``info["snapshot"]`` — the live
    :class:`~engine.GameSnapshot` (same object as ``_snapshot`` after the transition).
    On **terminal** steps only, ``info`` also contains ``combat_won`` (``True`` if the
    blind was beaten, ``False`` on terminal loss). Non-terminal steps and ``reset``
    return only ``snapshot``.

    **``shaping_gamma``:** PBRS coefficient in ``r + γ Φ(s') - Φ(s)``; set equal to
    the learner discount (e.g. ``make_vec(..., shaping_gamma=cfg.gamma)``) for the usual
    policy-invariant shaping.
    """

    metadata = {"render_modes": []}

    def __init__(self, snapshot: GameSnapshot, *, shaping_gamma: float = 1.0) -> None:
        super().__init__()
        self.observation_space = build_observation_space()
        self.action_space = spaces.Dict(
            {
                "selection": spaces.MultiBinary(MAX_HAND_LENGTH),
                "action_type": spaces.Discrete(2),
            }
        )
        self.shaping_gamma = float(shaping_gamma)
        self._init_snapshot_template: GameSnapshot = copy.deepcopy(snapshot)
        self._snapshot: GameSnapshot | None = None
        self._prev_potential: float = 0.0
        self._potential_w_flush: float = 0.0
        self._potential_w_three_kind: float = 0.0
        self._potential_w_straight: float = 0.0

    def _info(self, *, terminal: bool = False, combat_won: bool = False) -> dict:
        assert self._snapshot is not None
        out: dict = {"snapshot": self._snapshot}
        if terminal:
            out["combat_won"] = bool(combat_won)
        return out

    def _get_obs(self) -> dict:
        if self._snapshot is None:
            raise RuntimeError("internal error: _snapshot unset; call reset() first")
        return snapshot_to_obs_dict(self._snapshot)

    def _invalid_action_step(self) -> tuple[dict, float, bool, bool, dict]:
        """Invalid selection or illegal discard: ``INVALID_ACTION_REWARD`` plus PBRS, episode continues.

        The snapshot is unchanged, so ``Φ(s') = Φ(s)``; reuse :attr:`_prev_potential`
        (no :meth:`_state_potential` call). Shaping adds
        ``shaping_gamma * Φ(s) - Φ(s)`` — zero when ``shaping_gamma == 1``.
        """
        reward = float(INVALID_ACTION_REWARD)
        reward += self.shaping_gamma * self._prev_potential - self._prev_potential
        return self._get_obs(), reward, False, False, self._info()

    def _calculate_score(self, selected_cards: list[Card]) -> int:
        assert self._snapshot is not None
        return score_play(selected_cards, self._snapshot, self.np_random)

    def _state_potential(self, snapshot: GameSnapshot) -> float:
        """Potential Φ(s) for potential-based reward shaping (Ng et al.).

        On **valid** transitions :meth:`step` adds ``shaping_gamma * Φ(s') - Φ(s)``
        to the step reward. **Invalid** actions (see :meth:`_invalid_action_step`)
        add the same PBRS form with ``Φ(s') = Φ(s)`` using cached :attr:`_prev_potential`,
        plus ``INVALID_ACTION_REWARD``.

        Combines suit concentration (HHI of suit counts, ``Σ n_s²``), rank
        multimodality (HHI of rank counts, ``Σ n_r²``), and best 5-card straight
        *presence* window (squared max window sum over rank presence, with Ace high
        wrap). Card ids use ``suit = id // NUM_RANKS``, ``rank = id % NUM_RANKS``.

        ``chips × mult`` from :meth:`reset` weights: Flush (suit HHI), Three of a Kind
        (rank HHI / ``phi_sets``), Straight (straight window). Global scalars:
        ``STATE_POTENTIAL_W_*``. The weighted sum is divided by
        ``STATE_POTENTIAL_NORM`` so PBRS stays modest vs the win reward.
        """
        hand_card_ids = np.fromiter(
            (c.card_id for c in snapshot.hand), dtype=np.int32, count=len(snapshot.hand)
        )
        valid_ids = hand_card_ids[hand_card_ids >= 0]
        if len(valid_ids) == 0:
            return 0.0

        ranks = valid_ids % NUM_RANKS
        suits = valid_ids // NUM_RANKS

        suit_counts = np.bincount(suits, minlength=NUM_SUITS)
        phi_suit = float(np.sum(suit_counts**2))

        rank_counts = np.bincount(ranks, minlength=NUM_RANKS)
        phi_sets = float(np.sum(rank_counts**2))

        rank_present = (rank_counts > 0).astype(np.int32)
        padded_ranks = np.concatenate([rank_present, [rank_present[0]]])
        window = np.ones(5, dtype=np.int32)
        window_sums = np.convolve(padded_ranks, window, mode="valid")
        max_straight_len = int(np.max(window_sums))
        phi_straight = float(max_straight_len**2)

        raw = (
            STATE_POTENTIAL_W_SUIT * self._potential_w_flush * phi_suit
            + STATE_POTENTIAL_W_SET * self._potential_w_three_kind * phi_sets
            + STATE_POTENTIAL_W_STRAIGHT * self._potential_w_straight * phi_straight
        )
        return raw / float(STATE_POTENTIAL_NORM)

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
        self._potential_w_flush = _poker_hand_chips_times_mult(
            self._snapshot, HandType.FLUSH
        )
        self._potential_w_three_kind = _poker_hand_chips_times_mult(
            self._snapshot, HandType.THREE_OF_A_KIND
        )
        self._potential_w_straight = _poker_hand_chips_times_mult(
            self._snapshot, HandType.STRAIGHT
        )
        self._prev_potential = self._state_potential(self._snapshot)
        return self._get_obs(), self._info()

    def step(self, action):
        if self._snapshot is None:
            raise ResetNeeded("Call env.reset() before step().")
        selection = action["selection"]
        action_type = action["action_type"]
        snap = self._snapshot
        hand = snap.hand
        indices = _selected_indices(selection, len(hand))

        if _is_invalid_selection(indices):
            return self._invalid_action_step()

        if action_type == 1:
            if snap.play_remaining == 0:
                raise RuntimeError(
                    "cannot play: play_remaining is 0 (episode already ended)"
                )
        elif action_type == 0:
            if snap.discard_remaining == 0:
                return self._invalid_action_step()
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

        reached_target = snap.current_score >= snap.target_score
        terminated = reached_target or snap.play_remaining == 0
        if terminated:
            reward = (
                _terminal_reward(snap.play_remaining, snap.current_score)
                if reached_target
                else 0.0
            )
        else:
            _draw_until_hand_size(
                hand, snap.deck, snap.player_hand_size, self.np_random
            )
            reward = 0.0

        phi_prime = self._state_potential(snap)
        reward += self.shaping_gamma * phi_prime - self._prev_potential
        self._prev_potential = phi_prime

        return self._get_obs(), reward, terminated, False, self._info(
            terminal=terminated, combat_won=reached_target
        )
