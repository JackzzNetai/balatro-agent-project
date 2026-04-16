"""Card enhancement ids (``Card.enhancement`` and observation ``*_enhancements``).

Id ``0`` is **no** wiki enhancement (plain card). Wiki-aligned types follow
https://balatrowiki.org/w/Card_Modifiers#Enhancements — Bonus, Mult, Wild, Steel,
Lucky (subset of the full 8 in-game types). Ids are ``0``..``NUM_CARD_ENHANCEMENTS - 1``.

Prefer ``from defs import ...`` for the stable public API.
"""

from __future__ import annotations

from enum import IntEnum

NUM_CARD_ENHANCEMENTS = 6
CARD_ENHANCEMENT_HIGH = NUM_CARD_ENHANCEMENTS - 1


class CardEnhancement(IntEnum):
    """Integer enhancement type for playing cards."""

    NONE = 0
    BONUS = 1
    MULT = 2
    WILD = 3
    STEEL = 4
    LUCKY = 5


CARD_ENHANCEMENT_LABELS: dict[CardEnhancement, str] = {
    CardEnhancement.NONE: "Plain Card",
    CardEnhancement.BONUS: "Bonus Card",
    CardEnhancement.MULT: "Mult Card",
    CardEnhancement.WILD: "Wild Card",
    CardEnhancement.STEEL: "Steel Card",
    CardEnhancement.LUCKY: "Lucky Card",
}
