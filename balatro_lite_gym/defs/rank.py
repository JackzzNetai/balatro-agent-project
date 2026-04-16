"""Playing-card rank ids (``card_id % 13``).

Encoding matches ``engine.Card``: ``rank = card_id % NUM_RANKS`` with
``0`` = Ace through ``12`` = King. Same convention as poker logic in
``scoring`` and https://balatrowiki.org/w/Card_Ranks (Ace low for straights
where applicable; high-card ordering is separate in ``scoring``).

Prefer ``from defs import ...`` for the stable public API.
"""

from __future__ import annotations

from enum import IntEnum

NUM_RANKS = 13
RANK_HIGH = NUM_RANKS - 1


class CardRank(IntEnum):
    """Rank index within a suit (``card_id % NUM_RANKS``)."""

    ACE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    SEVEN = 6
    EIGHT = 7
    NINE = 8
    TEN = 9
    JACK = 10
    QUEEN = 11
    KING = 12


CARD_RANK_LABELS: dict[CardRank, str] = {
    CardRank.ACE: "A",
    CardRank.TWO: "2",
    CardRank.THREE: "3",
    CardRank.FOUR: "4",
    CardRank.FIVE: "5",
    CardRank.SIX: "6",
    CardRank.SEVEN: "7",
    CardRank.EIGHT: "8",
    CardRank.NINE: "9",
    CardRank.TEN: "T",
    CardRank.JACK: "J",
    CardRank.QUEEN: "Q",
    CardRank.KING: "K",
}
