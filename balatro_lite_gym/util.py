"""Card id and poker helpers (suit-major ``card_id``, recognition, ``played_contains``).

``card_id = suit * NUM_RANKS + rank`` — see :mod:`defs` for ``NUM_RANKS`` and enums.

:func:`rank_chips` is the wiki rank-to-chip table (also the in-game rank integer used
by e.g. Raised Fist). Poker classification follows https://balatrowiki.org/w/Poker_Hands
(including compound hands). ``played_contains`` encodes wiki **“hand contains X”**
(e.g. Three of a Kind contains a Pair), not “best classified type == X”. Wild
enhancement is respected for **contains** (rank flex + existing flush Wild
rules); :func:`recognize_poker_hand` classification still uses printed Wild ranks.
"""

from __future__ import annotations

from itertools import product

from defs import NUM_RANKS, CardEnhancement, CardRank, CardSuit, HandType
from defs.poker_hands import POKER_HAND_BASE_CHIPS_MULT, POKER_HAND_LEVEL_UP_CHIPS_MULT
from engine import Card


def rank_from_card_id(card_id: int) -> CardRank:
    """Rank from suit-major ``card_id`` (``0..51`` in a full deck)."""
    return CardRank(card_id % NUM_RANKS)


def suit_from_card_id(card_id: int) -> CardSuit:
    """Suit from suit-major ``card_id`` (``0..51`` in a full deck)."""
    return CardSuit(card_id // NUM_RANKS)


def card_id_from_suit_rank(suit: int, rank: int) -> int:
    """``card_id = suit * NUM_RANKS + rank`` (``suit`` and ``rank`` as ints)."""
    return suit * NUM_RANKS + rank


def card_rank(card_id: int) -> int:
    """Rank index 0..12 within a suit: 0 = Ace, 1 = 2, …, 12 = King (suit-major)."""
    return int(rank_from_card_id(card_id))


def card_suit(card_id: int) -> int:
    """Suit index 0..3: 0 clubs, 1 diamonds, 2 hearts, 3 spades."""
    return int(suit_from_card_id(card_id))


def rank_chips(card_rank: int) -> int:
    """In-game rank value: base chips when scored, same number Raised Fist uses ([Chips]).

    ``card_rank`` is ``card_id % NUM_RANKS``: 0=Ace … 8=nine, 9=ten, 10=J, 11=Q, 12=K.
    Ace ``11``, 2–9 map to face values ``2``..``9``, ten and faces ``10``.

    See https://balatrowiki.org/w/Chips and https://balatrowiki.org/w/Raised_Fist .
    """
    if not 0 <= card_rank < NUM_RANKS:
        raise ValueError(f"card_rank must be in 0..{NUM_RANKS - 1}, got {card_rank}")
    if card_rank == 0:
        return 11
    if 1 <= card_rank <= 8:
        return card_rank + 1
    return 10


def _rank_counts(ranks: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for r in ranks:
        out[r] = out.get(r, 0) + 1
    return out


def _flush_feasible(played: list[Card]) -> bool:
    """True if all non-Wild cards share one suit (Wild counts as every suit)."""
    anchor: int | None = None
    for c in played:
        if c.enhancement == CardEnhancement.WILD:
            continue
        s = card_suit(c.card_id)
        if anchor is None:
            anchor = s
        elif s != anchor:
            return False
    return True


def _all_same_rank(ranks: list[int]) -> bool:
    return bool(ranks) and len(set(ranks)) == 1


def _is_straight_ranks(ranks: list[int]) -> bool:
    """Five distinct ranks forming a straight (wheel or Broadway special-cased)."""
    if len(ranks) != 5:
        return False
    u = sorted(set(ranks))
    if len(u) != 5:
        return False
    if u == [0, 1, 2, 3, 4]:
        return True
    if set(u) == {0, 9, 10, 11, 12}:
        return True
    return u[-1] - u[0] == 4 and all(u[i + 1] - u[i] == 1 for i in range(4))


def _is_full_house_ranks(ranks: list[int]) -> bool:
    if len(ranks) != 5:
        return False
    vals = sorted(_rank_counts(ranks).values(), reverse=True)
    return vals == [3, 2]


def _is_two_pair_ranks(ranks: list[int]) -> bool:
    c = _rank_counts(ranks)
    if max(c.values(), default=0) >= 4:
        return False
    return sum(1 for v in c.values() if v == 2) == 2


def _is_three_of_a_kind_ranks(ranks: list[int]) -> bool:
    c = _rank_counts(ranks)
    mx = max(c.values())
    if mx != 3:
        return False
    return sorted(c.values(), reverse=True) != [3, 2]


def _is_one_pair_ranks(ranks: list[int]) -> bool:
    c = _rank_counts(ranks)
    if max(c.values()) != 2:
        return False
    return sum(1 for v in c.values() if v == 2) == 1


def _wild_split(played: list[Card]) -> tuple[list[Card], int]:
    """Non-Wild cards and Wild count (Wild = :attr:`~defs.CardEnhancement.WILD`)."""
    nw = [c for c in played if c.enhancement != CardEnhancement.WILD]
    return nw, len(played) - len(nw)


def _non_wild_rank_counts(played: list[Card]) -> dict[int, int]:
    nw, _ = _wild_split(played)
    return _rank_counts([card_rank(c.card_id) for c in nw])


def _max_rank_bucket_with_wilds(played: list[Card]) -> int:
    """Largest count any single rank can reach if all Wilds join that rank."""
    counts = _non_wild_rank_counts(played)
    _, w = _wild_split(played)
    return max(counts.values(), default=0) + w


_STRAIGHT_RANK_TEMPLATES: tuple[frozenset[int], ...] = (
    frozenset({0, 1, 2, 3, 4}),
    frozenset({0, 9, 10, 11, 12}),
    *(frozenset(range(low, low + 5)) for low in range(1, 9)),
)


def _wilds_needed_for_straight_template(
    template: frozenset[int], counts: dict[int, int]
) -> int:
    """Wilds required so each rank in ``template`` has at least one matching card."""
    need = 0
    for r in template:
        need += max(0, 1 - min(counts.get(r, 0), 1))
    return need


def _contains_straight_wild_aware(played: list[Card], n: int) -> bool:
    if n != 5:
        return False
    counts = _non_wild_rank_counts(played)
    _, w = _wild_split(played)
    return any(_wilds_needed_for_straight_template(t, counts) <= w for t in _STRAIGHT_RANK_TEMPLATES)


def _contains_two_pair_wild_aware(played: list[Card], n: int) -> bool:
    """Two Pair ``(2,2,1)`` multiset; Full House ``(3,2)`` does not qualify."""
    if n < 4 or n > 5:
        return False
    nw, w = _wild_split(played)
    base = [card_rank(c.card_id) for c in nw]
    if w == 0:
        return _is_two_pair_ranks([card_rank(c.card_id) for c in played])
    for extra in product(range(NUM_RANKS), repeat=w):
        ranks = base + list(extra)
        if _is_two_pair_ranks(ranks):
            return True
    return False


def _contains_full_house_wild_aware(played: list[Card], n: int) -> bool:
    if n != 5:
        return False
    counts = _non_wild_rank_counts(played)
    _, w = _wild_split(played)
    for a in range(NUM_RANKS):
        for b in range(NUM_RANKS):
            if a == b:
                continue
            if max(0, 3 - counts.get(a, 0)) + max(0, 2 - counts.get(b, 0)) <= w:
                return True
            if max(0, 2 - counts.get(a, 0)) + max(0, 3 - counts.get(b, 0)) <= w:
                return True
    return False


def _played_contains_high_card(played: list[Card], n: int) -> bool:
    """True iff no pair-or-better pattern (wiki sense) exists with Wild-aware ``contains``."""
    if n < 1:
        return False
    stronger = (
        HandType.STRAIGHT_FLUSH,
        HandType.FLUSH_FIVE,
        HandType.FLUSH_HOUSE,
        HandType.FIVE_OF_A_KIND,
        HandType.FOUR_OF_A_KIND,
        HandType.FULL_HOUSE,
        HandType.FLUSH,
        HandType.STRAIGHT,
        HandType.THREE_OF_A_KIND,
        HandType.TWO_PAIR,
        HandType.PAIR,
    )
    return not any(played_contains(played, h) for h in stronger)


def _ace_high_strength(rank: int) -> int:
    return 14 if rank == 0 else rank + 1


def _classify_poker_hand(played: list[Card], ranks: list[int]) -> HandType:
    """Hand type only; same precedence as ``recognize_poker_hand``."""
    n = len(played)
    if n == 5:
        if _all_same_rank(ranks):
            if _flush_feasible(played):
                return HandType.FLUSH_FIVE
            return HandType.FIVE_OF_A_KIND
        if _is_full_house_ranks(ranks) and _flush_feasible(played):
            return HandType.FLUSH_HOUSE
        if _is_straight_ranks(ranks) and _flush_feasible(played):
            return HandType.STRAIGHT_FLUSH

    if n >= 4 and max(_rank_counts(ranks).values(), default=0) >= 4:
        return HandType.FOUR_OF_A_KIND
    if n == 5 and _is_full_house_ranks(ranks):
        return HandType.FULL_HOUSE
    if n == 5 and _flush_feasible(played) and not _is_straight_ranks(ranks):
        return HandType.FLUSH
    if n == 5 and _is_straight_ranks(ranks) and not _flush_feasible(played):
        return HandType.STRAIGHT
    if _is_three_of_a_kind_ranks(ranks):
        return HandType.THREE_OF_A_KIND
    if _is_two_pair_ranks(ranks):
        return HandType.TWO_PAIR
    if _is_one_pair_ranks(ranks):
        return HandType.PAIR
    return HandType.HIGH_CARD


def _scored_indices(played: list[Card], ranks: list[int], hand: HandType) -> list[int]:
    """Indices into ``played`` for cards that score for this hand (wiki-relevant subset)."""
    n = len(played)
    counts = _rank_counts(ranks)

    if hand in (
        HandType.FLUSH_FIVE,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.STRAIGHT_FLUSH,
        HandType.FULL_HOUSE,
        HandType.FLUSH,
        HandType.STRAIGHT,
    ):
        return list(range(n))

    if hand == HandType.FOUR_OF_A_KIND:
        quad_rank = next(r for r, k in counts.items() if k >= 4)
        return [i for i in range(n) if ranks[i] == quad_rank]

    if hand == HandType.THREE_OF_A_KIND:
        trip_rank = next(r for r, k in counts.items() if k == 3)
        return [i for i in range(n) if ranks[i] == trip_rank]

    if hand == HandType.TWO_PAIR:
        pr = {r for r, k in counts.items() if k == 2}
        return [i for i in range(n) if ranks[i] in pr]

    if hand == HandType.PAIR:
        pair_rank = next(r for r, k in counts.items() if k == 2)
        return [i for i in range(n) if ranks[i] == pair_rank]

    if hand == HandType.HIGH_CARD:
        best = max(_ace_high_strength(r) for r in ranks)
        return [i for i in range(n) if _ace_high_strength(ranks[i]) == best]

    raise AssertionError(f"unhandled HandType: {hand}")


def recognize_poker_hand(
    played: list[Card], *, four_fingers: bool = False
) -> tuple[HandType, list[Card]]:
    """Classify the played subset (1–5 cards) and which cards score for that hand.

    Rules follow https://balatrowiki.org/w/Poker_Hands (including compound hands)
    and Wild enhancement from https://balatrowiki.org/w/Card_Modifiers
    (Wild = :attr:`~defs.CardEnhancement.WILD`: non-Wild suits must agree;
    Wild can match any).

    **Royal flush** is returned as :attr:`HandType.STRAIGHT_FLUSH`. With
    ``four_fingers=False`` (default), straights and flushes require five cards.

    ``played`` order does not affect classification (Balatro wiki). The second
    return value lists the **poker-relevant** played cards in **play order**
    (subset of ``played`` by reference); kickers are omitted where the wiki says
    they do not score. **High Card** uses **Ace high** among ``card_rank`` values.

    **Out of scope (ignored):** Splash (all played would score), Stone kickers,
    and other joker-driven membership changes.

    Returns:
        ``(hand_type, scored_cards)`` where ``scored_cards`` preserves the order
        of those cards within ``played``.
    """
    if four_fingers:
        raise NotImplementedError(
            "Four Fingers straight/flush rules are not implemented yet."
        )
    n = len(played)
    if n < 1 or n > 5:
        raise ValueError(f"played hand must have 1..5 cards, got {n}")
    ranks = [card_rank(c.card_id) for c in played]
    hand = _classify_poker_hand(played, ranks)
    idx = _scored_indices(played, ranks, hand)
    return hand, [played[i] for i in idx]


def played_contains(played: list[Card], inner: HandType) -> bool:
    """True if ``played`` (1–5 cards) **contains** poker pattern ``inner`` (wiki sense).

    **Contains** means the multiset admits the component (e.g. Three of a Kind
    **contains** a Pair). This is **not** the same as ``recognize_poker_hand(played)[0]
    == inner`` (e.g. Full House does not *contain* Two Pair in the two-pair sense).

    **Wild** enhancement: rank-based patterns treat Wilds as freely assignable rank
    copies (same spirit as :func:`_flush_feasible` for suits). **HIGH_CARD** is true
    only when no stronger ``contains`` pattern holds (so it stays consistent with
    ``recognize_poker_hand`` on typical hands; :func:`recognize_poker_hand` itself
    still uses printed Wild ranks for classification).

    Same five-card straight templates as :func:`_is_straight_ranks`; ``four_fingers``
    is not supported.

    Raises:
        ``ValueError`` if ``played`` is empty or longer than 5 cards.
    """
    n = len(played)
    if n < 1 or n > 5:
        raise ValueError(f"played hand must have 1..5 cards, got {n}")
    counts_no_wild = _non_wild_rank_counts(played)
    mx_wild = _max_rank_bucket_with_wilds(played)
    _, w = _wild_split(played)

    if inner is HandType.HIGH_CARD:
        return _played_contains_high_card(played, n)

    if inner is HandType.PAIR:
        return n >= 2 and mx_wild >= 2

    if inner is HandType.THREE_OF_A_KIND:
        return n >= 3 and mx_wild >= 3

    if inner is HandType.FOUR_OF_A_KIND:
        return n >= 4 and mx_wild >= 4

    if inner is HandType.TWO_PAIR:
        return _contains_two_pair_wild_aware(played, n)

    if inner is HandType.FULL_HOUSE:
        return _contains_full_house_wild_aware(played, n)

    if inner is HandType.STRAIGHT:
        return _contains_straight_wild_aware(played, n)

    if inner is HandType.FLUSH:
        return n == 5 and _flush_feasible(played)

    if inner is HandType.STRAIGHT_FLUSH:
        return (
            n == 5
            and _flush_feasible(played)
            and any(
                _wilds_needed_for_straight_template(t, counts_no_wild) <= w
                for t in _STRAIGHT_RANK_TEMPLATES
            )
        )

    if inner is HandType.FIVE_OF_A_KIND:
        return n == 5 and mx_wild >= 5 and not _flush_feasible(played)

    if inner is HandType.FLUSH_FIVE:
        return n == 5 and mx_wild >= 5 and _flush_feasible(played)

    if inner is HandType.FLUSH_HOUSE:
        return (
            n == 5
            and _flush_feasible(played)
            and _contains_full_house_wild_aware(played, n)
        )

    raise AssertionError(f"unhandled HandType for played_contains: {inner!r}")


def chips_mult_for_hand_level(hand: HandType, level: int) -> tuple[int, int]:
    """Wiki poker-hand line chips and **total** mult for ``hand`` at ``level`` (level 1 = base)."""
    if level < 1:
        raise ValueError(f"hand level must be >= 1, got {level}")
    bc, bm = POKER_HAND_BASE_CHIPS_MULT[hand]
    dc, dm = POKER_HAND_LEVEL_UP_CHIPS_MULT[hand]
    n = level - 1
    return bc + n * dc, bm + n * dm
