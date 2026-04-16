"""Joker scoring dispatch (activation-aligned subset).

Handlers match [List of Jokers](https://balatrowiki.org/w/Jokers#List_of_Jokers)
and timing from ``defs.jokers.JOKER_ACTIVATION``. ``joker.edition`` on jokers is
still ignored.

**Photograph:** round-local “first face scored” is not modeled on ``GameSnapshot``;
handler is a documented no-op until that state exists.

**Bloodstone / Misprint:** stochastic; require ``JokerEffectContext.rng`` when those
handlers run. ``rng`` is optional on the context for phases that never use it (e.g. On held).

**Wild** enhancement: suit-based **On scored** jokers treat Wild as matching **every**
suit (see :func:`_scored_card_counts_as_suit`). **Seeing Double** uses
:func:`_seeing_double_active` instead (Wild = one chosen suit at a time, wiki).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Final

import numpy as np

from defs import CardEnhancement, CardRank, HandType
from defs.jokers import JOKER_ACTIVATION, JokerActivation, JokerId, NUM_JOKERS
from defs.suit import CardSuit
from engine import Card, GameSnapshot, Joker
from util import card_rank, card_suit, played_contains, rank_chips

if TYPE_CHECKING:
    from scoring import ScoreAccumulator


_FIBONACCI_RANKS: Final[frozenset[CardRank]] = frozenset(
    {
        CardRank.ACE,
        CardRank.TWO,
        CardRank.THREE,
        CardRank.FIVE,
        CardRank.EIGHT,
    }
)
_EVEN_STEVEN_RANKS: Final[frozenset[CardRank]] = frozenset(
    {
        CardRank.TWO,
        CardRank.FOUR,
        CardRank.SIX,
        CardRank.EIGHT,
        CardRank.TEN,
    }
)
_ODD_TODD_RANKS: Final[frozenset[CardRank]] = frozenset(
    {
        CardRank.ACE,
        CardRank.THREE,
        CardRank.FIVE,
        CardRank.SEVEN,
        CardRank.NINE,
    }
)


# -----------------------------------------------------------------------------
# Context
# -----------------------------------------------------------------------------


@dataclass
class JokerEffectContext:
    """Arguments passed to each joker effect handler.

    ``rng`` may be omitted when no stochastic handler in that phase will run (e.g. On held).
    Stochastic jokers (Bloodstone, Misprint, …) require a non-``None`` ``rng``.
    """

    acc: ScoreAccumulator
    snapshot: GameSnapshot
    played: list[Card]
    scored_cards: list[Card]
    scored_card: Card | None = None
    held_card: Card | None = None
    rng: np.random.Generator | None = None


# -----------------------------------------------------------------------------
# Small rank / suit helpers (avoid importing ``scoring`` — circular import)
# -----------------------------------------------------------------------------


def _scored_card_counts_as_suit(card: Card, suit: int) -> bool:
    """True if ``card`` counts as ``suit`` for suit-based **On scored** jokers.

    **Wild** counts as **every** suit here (Greedy, Ancient, Bloodstone, etc.),
    matching poker Wild behavior in :mod:`util`. **Seeing Double** is different:
    it uses :func:`_seeing_double_active` (Wild = one chosen suit at a time).
    """
    if CardEnhancement(card.enhancement) is CardEnhancement.WILD:
        return True
    return card_suit(card.card_id) == suit


def _held_card_is_club_or_spade_or_wild(c: Card) -> bool:
    """Blackboard: Wild may count as Club or Spade (both black suits)."""
    if CardEnhancement(c.enhancement) is CardEnhancement.WILD:
        return True
    s = card_suit(c.card_id)
    return s in (int(CardSuit.CLUBS), int(CardSuit.SPADES))


def _played_covers_all_four_suits(cards: list[Card]) -> bool:
    """Flower Pot: can four suits be realized (non-wild suits + one suit per Wild)?"""
    if not cards:
        return False
    wilds = [c for c in cards if CardEnhancement(c.enhancement) is CardEnhancement.WILD]
    non_wild = [c for c in cards if CardEnhancement(c.enhancement) is not CardEnhancement.WILD]
    n_distinct = len({card_suit(c.card_id) for c in non_wild})
    return n_distinct + len(wilds) >= 4


def _even_steven_rank(card_id: int) -> bool:
    """Ranks 2, 4, 6, 8, 10 (wiki Even Steven)."""
    return CardRank(card_rank(card_id)) in _EVEN_STEVEN_RANKS


def _odd_todd_rank(card_id: int) -> bool:
    """Ranks A, 3, 5, 7, 9 (wiki Odd Todd)."""
    return CardRank(card_rank(card_id)) in _ODD_TODD_RANKS


def _is_face_rank(card_id: int) -> bool:
    """Jack, Queen, King (face cards for Scary Face / Smiley Face, etc.)."""
    return card_rank(card_id) >= CardRank.JACK


def _seeing_double_active(scored: list[Card]) -> bool:
    """Scoring Club and at least one other suit (wiki Seeing Double).

    **Wild** cards (:attr:`~defs.CardEnhancement.WILD`) are not one suit on the
    card id; each Wild may count as **one** suit of your choice. One Wild can
    supply the missing Club or the missing non-club if the other is already
    present among non-Wild scored cards; two Wilds alone can satisfy both.
    """
    if not scored:
        return False
    clubs = int(CardSuit.CLUBS)
    non_wild = [
        c for c in scored if CardEnhancement(c.enhancement) is not CardEnhancement.WILD
    ]
    wilds = [
        c for c in scored if CardEnhancement(c.enhancement) is CardEnhancement.WILD
    ]
    suits_nw = {card_suit(c.card_id) for c in non_wild}
    has_club_nw = clubs in suits_nw
    has_non_club_nw = any(s != clubs for s in suits_nw)

    if has_club_nw and has_non_club_nw:
        return True
    if not wilds:
        return False
    if has_club_nw or has_non_club_nw:
        return True
    return len(wilds) >= 2


# -----------------------------------------------------------------------------
# ON_SCORED
# -----------------------------------------------------------------------------


def _on_scored_suit_effect(
    ctx: JokerEffectContext,
    suit: int,
    value: float,
    is_multiplicative: bool,
) -> None:
    """If ``ctx.scored_card`` counts as ``suit``, apply ``value`` to Mult.

    ``is_multiplicative`` False: add ``value`` to Mult (sinful jokers). True: multiply
    Mult by ``value`` (Ancient).
    """
    if _scored_card_counts_as_suit(ctx.scored_card, suit):
        if is_multiplicative:
            ctx.acc.mult *= value
        else:
            ctx.acc.mult += value


def _effect_greedy_joker(ctx: JokerEffectContext) -> None:
    # Wiki: Played cards with Diamond suit give +3 Mult when scored.
    _on_scored_suit_effect(ctx, int(CardSuit.DIAMONDS), 3.0, False)


def _effect_lusty_joker(ctx: JokerEffectContext) -> None:
    # Wiki: Played cards with Heart suit give +3 Mult when scored.
    _on_scored_suit_effect(ctx, int(CardSuit.HEARTS), 3.0, False)


def _effect_wrathful_joker(ctx: JokerEffectContext) -> None:
    # Wiki: Played cards with Spade suit give +3 Mult when scored.
    _on_scored_suit_effect(ctx, int(CardSuit.SPADES), 3.0, False)


def _effect_gluttonous_joker(ctx: JokerEffectContext) -> None:
    # Wiki: Played cards with Club suit give +3 Mult when scored.
    _on_scored_suit_effect(ctx, int(CardSuit.CLUBS), 3.0, False)


def _effect_fibonacci(ctx: JokerEffectContext) -> None:
    # Wiki: Each played Ace, 2, 3, 5, or 8 gives +8 Mult when scored.
    if CardRank(card_rank(ctx.scored_card.card_id)) in _FIBONACCI_RANKS:
        ctx.acc.mult += 8.0


def _effect_scary_face(ctx: JokerEffectContext) -> None:
    # Wiki: Played face cards give +30 Chips when scored.
    if _is_face_rank(ctx.scored_card.card_id):
        ctx.acc.chips += 30


def _effect_even_steven(ctx: JokerEffectContext) -> None:
    # Wiki: Played cards with even rank give +4 Mult when scored (10, 8, 6, 4, 2).
    if _even_steven_rank(ctx.scored_card.card_id):
        ctx.acc.mult += 4.0


def _effect_odd_todd(ctx: JokerEffectContext) -> None:
    # Wiki: Played cards with odd rank give +31 Chips when scored (A, 9, 7, 5, 3).
    if _odd_todd_rank(ctx.scored_card.card_id):
        ctx.acc.chips += 31


def _effect_scholar(ctx: JokerEffectContext) -> None:
    # Wiki: Played Aces give +20 Chips and +4 Mult when scored.
    if CardRank(card_rank(ctx.scored_card.card_id)) == CardRank.ACE:
        ctx.acc.chips += 20
        ctx.acc.mult += 4.0


def _effect_ancient_clubs(ctx: JokerEffectContext) -> None:
    # Wiki (Ancient): X1.5 Mult per scored card of active suit; suit rotates in Balatro.
    # This JokerId pins Clubs.
    _on_scored_suit_effect(ctx, int(CardSuit.CLUBS), 1.5, True)


def _effect_ancient_diamonds(ctx: JokerEffectContext) -> None:
    # Wiki: Ancient Joker — this JokerId pins Diamonds.
    _on_scored_suit_effect(ctx, int(CardSuit.DIAMONDS), 1.5, True)


def _effect_ancient_hearts(ctx: JokerEffectContext) -> None:
    # Wiki: Ancient Joker — this JokerId pins Hearts.
    _on_scored_suit_effect(ctx, int(CardSuit.HEARTS), 1.5, True)


def _effect_ancient_spades(ctx: JokerEffectContext) -> None:
    # Wiki: Ancient Joker — this JokerId pins Spades.
    _on_scored_suit_effect(ctx, int(CardSuit.SPADES), 1.5, True)


def _effect_walkie_talkie(ctx: JokerEffectContext) -> None:
    # Wiki: Each played 10 and 4 give +10 Chips and +4 Mult when scored.
    r = CardRank(card_rank(ctx.scored_card.card_id))
    if r in (CardRank.FOUR, CardRank.TEN):
        ctx.acc.chips += 10
        ctx.acc.mult += 4.0


def _effect_smiley_face(ctx: JokerEffectContext) -> None:
    # Wiki: Played face cards give +5 Mult when scored.
    if _is_face_rank(ctx.scored_card.card_id):
        ctx.acc.mult += 5.0


def _effect_bloodstone(ctx: JokerEffectContext) -> None:
    # Wiki: 1 in 2 chance for played Heart cards to give X1.5 Mult when scored.
    if ctx.rng is None:
        raise TypeError("Bloodstone requires JokerEffectContext.rng")
    if _scored_card_counts_as_suit(
        ctx.scored_card, int(CardSuit.HEARTS)
    ) and ctx.rng.random() < 0.5:
        ctx.acc.mult *= 1.5


def _effect_arrowhead(ctx: JokerEffectContext) -> None:
    # Wiki: Played Spade cards give +50 Chips when scored.
    if _scored_card_counts_as_suit(ctx.scored_card, int(CardSuit.SPADES)):
        ctx.acc.chips += 50


def _effect_onyx_agate(ctx: JokerEffectContext) -> None:
    # Wiki: Played Club cards give +7 Mult when scored.
    if _scored_card_counts_as_suit(ctx.scored_card, int(CardSuit.CLUBS)):
        ctx.acc.mult += 7.0


def _effect_photograph(ctx: JokerEffectContext) -> None:
    # Wiki: First played face card gives X2 Mult when scored (not implemented here).
    _ = ctx


def _effect_triboulet(ctx: JokerEffectContext) -> None:
    # Wiki: Played Kings and Queens each give X2 Mult when scored.
    r = CardRank(card_rank(ctx.scored_card.card_id))
    if r in (CardRank.QUEEN, CardRank.KING):
        ctx.acc.mult *= 2.0


# -----------------------------------------------------------------------------
# ON_HELD
# -----------------------------------------------------------------------------


def _effect_raised_fist(ctx: JokerEffectContext) -> None:
    # Wiki: Adds 2× the rank of the lowest rank card held in hand to Mult.
    # ON_HELD runs once per held card × joker; only hand[0] applies so the bonus is once per play.
    hand = ctx.snapshot.hand
    if hand and ctx.held_card is hand[0]:
        min_rank = min(card_rank(c.card_id) for c in hand)
        ctx.acc.mult += 2.0 * float(rank_chips(min_rank))


def _effect_baron(ctx: JokerEffectContext) -> None:
    # Wiki: Each King held in hand gives X1.5 Mult.
    if CardRank(card_rank(ctx.held_card.card_id)) == CardRank.KING:
        ctx.acc.mult *= 1.5


def _effect_shoot_the_moon(ctx: JokerEffectContext) -> None:
    # Wiki: Each Queen held in hand gives +13 Mult.
    if CardRank(card_rank(ctx.held_card.card_id)) == CardRank.QUEEN:
        ctx.acc.mult += 13.0


# -----------------------------------------------------------------------------
# INDEPENDENT
# -----------------------------------------------------------------------------


def _effect_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +4 Mult.
    ctx.acc.mult += 4.0


def _effect_jolly_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +8 Mult if played hand contains a Pair.
    if played_contains(ctx.played, HandType.PAIR):
        ctx.acc.mult += 8.0


def _effect_zany_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +12 Mult if played hand contains a Three of a Kind.
    if played_contains(ctx.played, HandType.THREE_OF_A_KIND):
        ctx.acc.mult += 12.0


def _effect_mad_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +10 Mult if played hand contains a Two Pair.
    if played_contains(ctx.played, HandType.TWO_PAIR):
        ctx.acc.mult += 10.0


def _effect_crazy_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +12 Mult if played hand contains a Straight.
    if played_contains(ctx.played, HandType.STRAIGHT):
        ctx.acc.mult += 12.0


def _effect_droll_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +10 Mult if played hand contains a Flush.
    if played_contains(ctx.played, HandType.FLUSH):
        ctx.acc.mult += 10.0


def _effect_sly_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +50 Chips if played hand contains a Pair.
    if played_contains(ctx.played, HandType.PAIR):
        ctx.acc.chips += 50


def _effect_wily_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +100 Chips if played hand contains a Three of a Kind.
    if played_contains(ctx.played, HandType.THREE_OF_A_KIND):
        ctx.acc.chips += 100


def _effect_clever_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +80 Chips if played hand contains a Two Pair.
    if played_contains(ctx.played, HandType.TWO_PAIR):
        ctx.acc.chips += 80


def _effect_devious_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +100 Chips if played hand contains a Straight.
    if played_contains(ctx.played, HandType.STRAIGHT):
        ctx.acc.chips += 100


def _effect_crafty_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +80 Chips if played hand contains a Flush.
    if played_contains(ctx.played, HandType.FLUSH):
        ctx.acc.chips += 80


def _effect_half_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +20 Mult if played hand contains 3 or fewer cards.
    if len(ctx.played) <= 3:
        ctx.acc.mult += 20.0


def _effect_banner(ctx: JokerEffectContext) -> None:
    # Wiki: +30 Chips for each discard remaining.
    ctx.acc.chips += 30 * ctx.snapshot.discard_remaining


def _effect_mystic_summit(ctx: JokerEffectContext) -> None:
    # Wiki: +15 Mult when 0 discards remaining.
    if ctx.snapshot.discard_remaining == 0:
        ctx.acc.mult += 15.0


def _effect_misprint(ctx: JokerEffectContext) -> None:
    # Wiki: +0–23 Mult (random each time scored).
    if ctx.rng is None:
        raise TypeError("Misprint requires JokerEffectContext.rng")
    ctx.acc.mult += int(ctx.rng.integers(0, 24))


def _effect_blackboard(ctx: JokerEffectContext) -> None:
    # Wiki: X3 Mult if every card left in hand (after the play) is Spade or Club; also
    # triggers when hand is empty (“no cards left”).
    hand = ctx.snapshot.hand
    if not hand or all(_held_card_is_club_or_spade_or_wild(c) for c in hand):
        ctx.acc.mult *= 3.0


def _effect_blue_joker(ctx: JokerEffectContext) -> None:
    # Wiki: +2 Chips for each remaining card in deck.
    ctx.acc.chips += 2 * len(ctx.snapshot.deck)


def _effect_flower_pot(ctx: JokerEffectContext) -> None:
    # Wiki: X3 Mult if poker hand contains Diamond, Club, Heart and Spade cards.
    if _played_covers_all_four_suits(ctx.played):
        ctx.acc.mult *= 3.0


def _effect_seeing_double(ctx: JokerEffectContext) -> None:
    # Wiki: X2 Mult if played hand has a scoring Club card and a scoring card of any other suit.
    if _seeing_double_active(ctx.scored_cards):
        ctx.acc.mult *= 2.0


def _effect_the_duo(ctx: JokerEffectContext) -> None:
    # Wiki: X2 Mult if played hand contains a Pair.
    if played_contains(ctx.played, HandType.PAIR):
        ctx.acc.mult *= 2.0


def _effect_the_trio(ctx: JokerEffectContext) -> None:
    # Wiki: X3 Mult if played hand contains a Three of a Kind.
    if played_contains(ctx.played, HandType.THREE_OF_A_KIND):
        ctx.acc.mult *= 3.0


def _effect_the_family(ctx: JokerEffectContext) -> None:
    # Wiki: X4 Mult if played hand contains a Four of a Kind.
    if played_contains(ctx.played, HandType.FOUR_OF_A_KIND):
        ctx.acc.mult *= 4.0


def _effect_the_order(ctx: JokerEffectContext) -> None:
    # Wiki: X3 Mult if played hand contains a Straight.
    if played_contains(ctx.played, HandType.STRAIGHT):
        ctx.acc.mult *= 3.0


def _effect_the_tribe(ctx: JokerEffectContext) -> None:
    # Wiki: X2 Mult if played hand contains a Flush.
    if played_contains(ctx.played, HandType.FLUSH):
        ctx.acc.mult *= 2.0


def _effect_acrobat(ctx: JokerEffectContext) -> None:
    # Wiki: X3 Mult on final hand of round. Env decrements play_remaining before scoring,
    # so the last play sees play_remaining == 0.
    if ctx.snapshot.play_remaining == 0:
        ctx.acc.mult *= 3.0


# -----------------------------------------------------------------------------
# Registry (wiki Nr traceability in ``defs/jokers.py`` module docstring)
# -----------------------------------------------------------------------------


def _build_effect_handlers() -> dict[JokerId, Callable[[JokerEffectContext], None]]:
    return {
        JokerId.JOKER: _effect_joker,
        JokerId.GREEDY_JOKER: _effect_greedy_joker,
        JokerId.LUSTY_JOKER: _effect_lusty_joker,
        JokerId.WRATHFUL_JOKER: _effect_wrathful_joker,
        JokerId.GLUTTONOUS_JOKER: _effect_gluttonous_joker,
        JokerId.JOLLY_JOKER: _effect_jolly_joker,
        JokerId.ZANY_JOKER: _effect_zany_joker,
        JokerId.MAD_JOKER: _effect_mad_joker,
        JokerId.CRAZY_JOKER: _effect_crazy_joker,
        JokerId.DROLL_JOKER: _effect_droll_joker,
        JokerId.SLY_JOKER: _effect_sly_joker,
        JokerId.WILY_JOKER: _effect_wily_joker,
        JokerId.CLEVER_JOKER: _effect_clever_joker,
        JokerId.DEVIOUS_JOKER: _effect_devious_joker,
        JokerId.CRAFTY_JOKER: _effect_crafty_joker,
        JokerId.HALF_JOKER: _effect_half_joker,
        JokerId.BANNER: _effect_banner,
        JokerId.MYSTIC_SUMMIT: _effect_mystic_summit,
        JokerId.MISPRINT: _effect_misprint,
        JokerId.RAISED_FIST: _effect_raised_fist,
        JokerId.FIBONACCI: _effect_fibonacci,
        JokerId.SCARY_FACE: _effect_scary_face,
        JokerId.EVEN_STEVEN: _effect_even_steven,
        JokerId.ODD_TODD: _effect_odd_todd,
        JokerId.SCHOLAR: _effect_scholar,
        JokerId.BLACKBOARD: _effect_blackboard,
        JokerId.BLUE_JOKER: _effect_blue_joker,
        JokerId.BARON: _effect_baron,
        JokerId.PHOTOGRAPH: _effect_photograph,
        JokerId.ANCIENT_JOKER_CLUBS: _effect_ancient_clubs,
        JokerId.ANCIENT_JOKER_DIAMONDS: _effect_ancient_diamonds,
        JokerId.ANCIENT_JOKER_HEARTS: _effect_ancient_hearts,
        JokerId.ANCIENT_JOKER_SPADES: _effect_ancient_spades,
        JokerId.WALKIE_TALKIE: _effect_walkie_talkie,
        JokerId.SMILEY_FACE: _effect_smiley_face,
        JokerId.ACROBAT: _effect_acrobat,
        JokerId.BLOODSTONE: _effect_bloodstone,
        JokerId.ARROWHEAD: _effect_arrowhead,
        JokerId.ONYX_AGATE: _effect_onyx_agate,
        JokerId.FLOWER_POT: _effect_flower_pot,
        JokerId.SEEING_DOUBLE: _effect_seeing_double,
        JokerId.THE_DUO: _effect_the_duo,
        JokerId.THE_TRIO: _effect_the_trio,
        JokerId.THE_FAMILY: _effect_the_family,
        JokerId.THE_ORDER: _effect_the_order,
        JokerId.THE_TRIBE: _effect_the_tribe,
        JokerId.SHOOT_THE_MOON: _effect_shoot_the_moon,
        JokerId.TRIBOULET: _effect_triboulet,
    }


EFFECT_HANDLERS: Final[dict[JokerId, Callable[[JokerEffectContext], None]]] = (
    _build_effect_handlers()
)


def _verify_handlers_dense() -> None:
    for jid in JokerId:
        if JOKER_ACTIVATION[jid] is None:
            continue
        if jid not in EFFECT_HANDLERS:
            raise AssertionError(f"Missing EFFECT_HANDLERS for scoring joker {jid!r}")


_verify_handlers_dense()


def try_applying_joker_effect(
    curr_activation: JokerActivation,
    joker: Joker,
    *,
    ctx: JokerEffectContext,
) -> None:
    """Invoke the registered handler if this joker scores in ``curr_activation``.

    ``joker.edition`` is ignored until joker editions are implemented.
    """
    if curr_activation is None:
        raise TypeError("curr_activation must be a JokerActivation member, not None")
    jid_int = int(joker.id)
    if not (0 <= jid_int < NUM_JOKERS):
        raise ValueError(f"joker.id must be in 0..{NUM_JOKERS - 1}, got {joker.id!r}")
    jid = JokerId(jid_int)
    expected = JOKER_ACTIVATION[jid]
    if curr_activation != expected:
        return
    fn = EFFECT_HANDLERS.get(jid)
    if fn is None:
        raise KeyError(f"Missing EFFECT_HANDLERS entry for scoring joker {jid!r}")
    fn(ctx)
