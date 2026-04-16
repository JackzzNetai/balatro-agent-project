"""Playing-card suit ids (``card_id // 13``).

Encoding matches ``engine.Card``: ``suit = card_id // NUM_RANKS`` with
``rank`` from :mod:`defs.rank`. Order is **clubs, diamonds, hearts, spades** —
``0``..``3`` — consistent with ``util.card_suit`` and wiki suit names
https://balatrowiki.org/w/Card_Suits .

Prefer ``from defs import ...`` for the stable public API.
"""

from __future__ import annotations

from enum import IntEnum

from .rank import NUM_RANKS

NUM_SUITS = 4
SUIT_HIGH = NUM_SUITS - 1


class CardSuit(IntEnum):
    """Suit index in suit-major ``card_id`` (``card_id // NUM_RANKS``)."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


CARD_SUIT_LABELS: dict[CardSuit, str] = {
    CardSuit.CLUBS: "Clubs",
    CardSuit.DIAMONDS: "Diamonds",
    CardSuit.HEARTS: "Hearts",
    CardSuit.SPADES: "Spades",
}
