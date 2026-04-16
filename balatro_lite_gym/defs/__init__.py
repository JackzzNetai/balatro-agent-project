"""Balatro-lite id tables: ranks, suits, enhancements, editions, poker hands, jokers, bosses.

Import from here for the stable public API, e.g.
``from defs import HandType, CardRank, CardEnhancement, Edition, JokerId, BossBlind, BOSS_BLIND_LABELS, NO_BOSS_BLIND_ID``.
"""

from __future__ import annotations

from .bosses import BOSS_BLIND_LABELS, BossBlind, NO_BOSS_BLIND_ID
from .editions import (
    CARD_EDITION_HIGH,
    EDITION_LABELS,
    Edition,
    JOKER_EDITION_HIGH,
    NUM_CARD_EDITIONS,
    NUM_JOKER_EDITIONS,
)
from .enhancements import (
    CARD_ENHANCEMENT_HIGH,
    CARD_ENHANCEMENT_LABELS,
    CardEnhancement,
    NUM_CARD_ENHANCEMENTS,
)
from .jokers import (
    JOKER_ACTIVATION,
    JOKER_ID_HIGH,
    JOKER_LABELS,
    JokerActivation,
    JokerId,
    NUM_JOKERS,
)
from .poker_hands import HAND_TYPE_COUNT, HAND_TYPE_LABELS, HandType
from .rank import CARD_RANK_LABELS, NUM_RANKS, RANK_HIGH, CardRank
from .suit import CARD_SUIT_LABELS, NUM_SUITS, SUIT_HIGH, CardSuit

__all__ = [
    "BOSS_BLIND_LABELS",
    "BossBlind",
    "NO_BOSS_BLIND_ID",
    "CARD_EDITION_HIGH",
    "CARD_ENHANCEMENT_HIGH",
    "CARD_ENHANCEMENT_LABELS",
    "CARD_RANK_LABELS",
    "CARD_SUIT_LABELS",
    "CardEnhancement",
    "CardRank",
    "CardSuit",
    "Edition",
    "HAND_TYPE_COUNT",
    "HAND_TYPE_LABELS",
    "HandType",
    "JOKER_ACTIVATION",
    "JOKER_EDITION_HIGH",
    "JOKER_ID_HIGH",
    "JOKER_LABELS",
    "JokerActivation",
    "JokerId",
    "NUM_JOKERS",
    "NUM_CARD_EDITIONS",
    "NUM_CARD_ENHANCEMENTS",
    "NUM_JOKER_EDITIONS",
    "NUM_RANKS",
    "NUM_SUITS",
    "RANK_HIGH",
    "SUIT_HIGH",
    "EDITION_LABELS",
]
