"""Card and joker edition ids (``Card.edition``, ``Joker.edition``, observation ``*_editions``).

Order matches ``environment`` / the wiki edition list
https://balatrowiki.org/w/Card_Modifiers#Editions — **Base**, **Foil**,
**Holographic**, **Polychrome** (subset; wiki also lists **Negative**). Ids are
``0``..``NUM_CARD_EDITIONS - 1``. Jokers use the **same** id set and count as
these playing-card editions in ``environment``.

Prefer ``from defs import ...`` for the stable public API.
"""

from __future__ import annotations

from enum import IntEnum

NUM_CARD_EDITIONS = 4
NUM_JOKER_EDITIONS = NUM_CARD_EDITIONS
CARD_EDITION_HIGH = NUM_CARD_EDITIONS - 1
JOKER_EDITION_HIGH = NUM_JOKER_EDITIONS - 1


class Edition(IntEnum):
    """Integer edition for cards and jokers."""

    BASE = 0
    FOIL = 1
    HOLOGRAPHIC = 2
    POLYCHROME = 3


EDITION_LABELS: dict[Edition, str] = {
    Edition.BASE: "Base",
    Edition.FOIL: "Foil",
    Edition.HOLOGRAPHIC: "Holographic",
    Edition.POLYCHROME: "Polychrome",
}
