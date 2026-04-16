"""Play scoring: played cards + snapshot → score delta for one **play** action.

Only a **subset** of jokers / rules will be modeled. **Pre-scoring** (dealt-hand
mutators like Vampire / Midas before the visible scoring pass) is **out of
scope** for now. The pipeline follows a subset of the wiki **Activation
Sequence**, in order:

- **On scored** — poker ``hand_levels[hand]`` as ``[chips, mult]``, then each
  **scored** card: rank chips ([Chips](https://balatrowiki.org/w/Chips)),
  enhancements, editions (Foil → Holo → Polychrome), then **On scored** jokers
  left-to-right per scored card (via :func:`joker_effects.try_applying_joker_effect`).
- **On held** — each card in ``snapshot.hand`` left-to-right, and for each card
  **On held** jokers left-to-right.
- **Independent** — ``snapshot.jokers`` left-to-right (joker editions deferred).

The environment calls :func:`score_play`; discards do not use this module.

**Ordering:** ``selected_cards``, ``snapshot.jokers``, and ``snapshot.hand`` are
assumed **already in evaluation order** (e.g. left-to-right as laid out in the
UI). This module does **not** sort or reorder them.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from defs import CardEnhancement, Edition, HandType, JokerActivation
from engine import Card, GameSnapshot
from joker_effects import JokerEffectContext, try_applying_joker_effect
from util import card_rank, rank_chips, recognize_poker_hand


@dataclass
class ScoreAccumulator:
    """Balatro-style line: final play contribution is ``chips * mult`` (then int)."""

    chips: int = 0
    mult: float = 1.0

    def total(self) -> int:
        return int(self.chips * self.mult)


def _hand_line_chips_mult(snapshot: GameSnapshot, hand: HandType) -> tuple[int, float]:
    key = int(hand)
    if key not in snapshot.hand_levels:
        raise KeyError(
            f"hand_levels missing key for classified hand {hand!s} (int {key}); "
            "expected snapshot.hand_levels[int(hand)] = [chips, mult]"
        )
    pair = snapshot.hand_levels[key]
    if len(pair) != 2:
        raise ValueError(
            f"hand_levels[{key}] must be a length-2 [chips, mult] list, got {pair!r}"
        )
    return int(pair[0]), float(pair[1])


def _accumulate_on_scored_card_intrinsics(
    c: Card, acc: ScoreAccumulator, rng: Generator
) -> None:
    """Rank chips, enhancement, and edition for one scored card (before On scored jokers)."""
    acc.chips += rank_chips(card_rank(c.card_id))
    enh = CardEnhancement(c.enhancement)
    if enh is CardEnhancement.BONUS:
        acc.chips += 30
    elif enh is CardEnhancement.MULT:
        acc.mult += 4.0
    elif enh is CardEnhancement.LUCKY:
        if rng.random() < 0.2:
            acc.mult += 20.0
    elif enh in (CardEnhancement.NONE, CardEnhancement.WILD, CardEnhancement.STEEL):
        pass
    else:
        raise AssertionError(f"unhandled CardEnhancement: {enh!r}")

    ed = Edition(c.edition)
    if ed is Edition.BASE:
        pass
    elif ed is Edition.FOIL:
        acc.chips += 50
    elif ed is Edition.HOLOGRAPHIC:
        acc.mult += 10.0
    elif ed is Edition.POLYCHROME:
        acc.mult *= 1.5
    else:
        raise AssertionError(f"unhandled Edition: {ed!r}")


def _scoring_on_scored(
    curr_activation: JokerActivation,
    played: list[Card],
    snapshot: GameSnapshot,
    acc: ScoreAccumulator,
    rng: Generator,
    hand: HandType,
    scored: list[Card],
) -> None:
    """Dealt-hand line: poker ``[chips, mult]`` then each scored card and its jokers.

    Order: add ``hand_levels[hand]`` pair → for each scored card: rank chips,
    enhancements (Bonus / Mult / Lucky partial; no Steel), editions Foil → Holo
    → Polychrome → jokers at ``curr_activation`` L→R.
    """
    hc, hm = _hand_line_chips_mult(snapshot, hand)
    acc.chips += hc
    acc.mult += hm
    for c in scored:
        _accumulate_on_scored_card_intrinsics(c, acc, rng)
        for joker in snapshot.jokers:
            try_applying_joker_effect(
                curr_activation,
                joker,
                ctx=JokerEffectContext(
                    acc=acc,
                    snapshot=snapshot,
                    played=played,
                    scored_cards=scored,
                    scored_card=c,
                    held_card=None,
                    rng=rng,
                ),
            )


def _scoring_on_held(
    curr_activation: JokerActivation,
    snapshot: GameSnapshot,
    acc: ScoreAccumulator,
) -> None:
    """``snapshot.hand`` × ``snapshot.jokers`` at ``curr_activation`` (On held, wiki order)."""
    for held in snapshot.hand:
        for joker in snapshot.jokers:
            try_applying_joker_effect(
                curr_activation,
                joker,
                ctx=JokerEffectContext(
                    acc=acc,
                    snapshot=snapshot,
                    played=[],
                    scored_cards=[],
                    scored_card=None,
                    held_card=held,
                ),
            )


def _scoring_independent(
    curr_activation: JokerActivation,
    played: list[Card],
    snapshot: GameSnapshot,
    acc: ScoreAccumulator,
    rng: Generator,
    scored: list[Card],
) -> None:
    """Jokers at ``curr_activation`` (Independent) L→R (joker editions deferred)."""
    for joker in snapshot.jokers:
        try_applying_joker_effect(
            curr_activation,
            joker,
            ctx=JokerEffectContext(
                acc=acc,
                snapshot=snapshot,
                played=played,
                scored_cards=scored,
                scored_card=None,
                held_card=None,
                rng=rng,
            ),
        )


def score_play(
    played_cards: list[Card],
    snapshot: GameSnapshot,
    rng: Generator,
) -> int:
    """Score one play. No pre-scoring pass (subset of jokers).

    ``played_cards`` is in play order (caller-maintained; not re-sorted here).
    ``snapshot`` is **after** those cards are removed from ``hand`` (the held
    hand during scoring, before any post-score draw). The gym
    :class:`~environment.BalatroLiteEnv` decrements ``play_remaining`` **before**
    calling this function, so the final play of a round has ``play_remaining == 0``.
    ``current_score`` is still the pre-play value here. ``jokers`` / ``hand_levels`` /
    ``blind_id`` match what you need for **On scored**, **On held**, and **Independent**.
    ``snapshot.jokers`` and ``snapshot.hand`` are assumed already in evaluation order.

    ``rng`` is **required** (e.g. ``env.np_random``) for stochastic effects such
    as Lucky Card; ``None`` is not allowed.
    """
    if rng is None:
        raise TypeError("score_play requires a numpy.random.Generator as rng=, not None")
    acc = ScoreAccumulator()
    hand, scored = recognize_poker_hand(played_cards)
    curr_activation = JokerActivation.ON_SCORED
    _scoring_on_scored(curr_activation, played_cards, snapshot, acc, rng, hand, scored)
    curr_activation = JokerActivation.ON_HELD
    _scoring_on_held(curr_activation, snapshot, acc)
    curr_activation = JokerActivation.INDEPENDENT
    _scoring_independent(curr_activation, played_cards, snapshot, acc, rng, scored)
    return acc.total()
