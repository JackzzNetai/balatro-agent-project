"""Balatro joker ids for this gym (dense ``JokerId`` 0..54).

Wiki cross-reference (Balatro Wiki *Nr* column, see
https://balatrowiki.org/w/Jokers#List_of_Jokers). ``JokerId`` is **not** the wiki
row number; comments show the wiki *Nr* for traceability.

Ordering: all selected wiki rows with *Nr* < 99 ascending; then four Ancient
variants (wiki *Nr* 99) in :class:`suit.CardSuit` order (Clubs..Spades); then
all selected wiki rows with *Nr* > 99 ascending.

JokerId  : wiki Nr
---------+--------
0..32    : 1,2,...,16,18,22,23,27,29,31,33,37,39,40,41,48,52,53,69,72,78
33..36   : 99 (Ancient ×4 by suit: Clubs, Diamonds, Hearts, Spades)
37..54   : 101,104,108,113,117,118,119,122,123,128,131,132,133,134,135,138,140,147

Total **55** jokers = 51 wiki-backed selections + 4 Ancient suit splits.
"""

from __future__ import annotations

from enum import Enum, IntEnum

NUM_JOKERS = 55
JOKER_ID_HIGH = NUM_JOKERS - 1


class JokerActivation(Enum):
    """When a joker contributes to play score (wiki activation sequence subset)."""

    ON_SCORED = "on_scored"
    ON_HELD = "on_held"
    INDEPENDENT = "independent"


class JokerId(IntEnum):
    """Dense gym joker id (see module docstring for wiki *Nr* mapping)."""

    # wiki 1–16
    JOKER = 0
    GREEDY_JOKER = 1
    LUSTY_JOKER = 2
    WRATHFUL_JOKER = 3
    GLUTTONOUS_JOKER = 4
    JOLLY_JOKER = 5
    ZANY_JOKER = 6
    MAD_JOKER = 7
    CRAZY_JOKER = 8
    DROLL_JOKER = 9
    SLY_JOKER = 10
    WILY_JOKER = 11
    CLEVER_JOKER = 12
    DEVIOUS_JOKER = 13
    CRAFTY_JOKER = 14
    HALF_JOKER = 15
    # wiki 18, 22, 23, 27, 29, 31, 33, 37, 39, 40, 41, 48, 52, 53, 69, 72, 78
    FOUR_FINGERS = 16
    BANNER = 17
    MYSTIC_SUMMIT = 18
    MISPRINT = 19
    RAISED_FIST = 20
    FIBONACCI = 21
    SCARY_FACE = 22
    PAREIDOLIA = 23
    EVEN_STEVEN = 24
    ODD_TODD = 25
    SCHOLAR = 26
    BLACKBOARD = 27
    SPLASH = 28
    BLUE_JOKER = 29
    SHORTCUT = 30
    BARON = 31
    PHOTOGRAPH = 32
    # wiki 99 — suit-fixed gym variants (wiki Ancient Joker rotates suit)
    ANCIENT_JOKER_CLUBS = 33
    ANCIENT_JOKER_DIAMONDS = 34
    ANCIENT_JOKER_HEARTS = 35
    ANCIENT_JOKER_SPADES = 36
    # wiki 101, 104, 108, 113, 117, 118, 119, 122, 123, 128, 131–135, 138, 140, 147
    WALKIE_TALKIE = 37
    SMILEY_FACE = 38
    ACROBAT = 39
    SMEARED_JOKER = 40
    BLOODSTONE = 41
    ARROWHEAD = 42
    ONYX_AGATE = 43
    FLOWER_POT = 44
    BLUEPRINT = 45
    SEEING_DOUBLE = 46
    THE_DUO = 47
    THE_TRIO = 48
    THE_FAMILY = 49
    THE_ORDER = 50
    THE_TRIBE = 51
    BRAINSTORM = 52
    SHOOT_THE_MOON = 53
    TRIBOULET = 54


_JA = JokerActivation

# Dense: one entry per JokerId. ``None`` = non-scoring / passive for this gym (no
# ``try_applying`` handler). Phases follow https://balatrowiki.org/w/Activation_Type
# (On scored / On held / Independent); Blueprint / Brainstorm are ``None`` here
# until copying is modeled.
JOKER_ACTIVATION: dict[JokerId, JokerActivation | None] = {
    JokerId.JOKER: _JA.INDEPENDENT,
    JokerId.GREEDY_JOKER: _JA.ON_SCORED,
    JokerId.LUSTY_JOKER: _JA.ON_SCORED,
    JokerId.WRATHFUL_JOKER: _JA.ON_SCORED,
    JokerId.GLUTTONOUS_JOKER: _JA.ON_SCORED,
    JokerId.JOLLY_JOKER: _JA.INDEPENDENT,
    JokerId.ZANY_JOKER: _JA.INDEPENDENT,
    JokerId.MAD_JOKER: _JA.INDEPENDENT,
    JokerId.CRAZY_JOKER: _JA.INDEPENDENT,
    JokerId.DROLL_JOKER: _JA.INDEPENDENT,
    JokerId.SLY_JOKER: _JA.INDEPENDENT,
    JokerId.WILY_JOKER: _JA.INDEPENDENT,
    JokerId.CLEVER_JOKER: _JA.INDEPENDENT,
    JokerId.DEVIOUS_JOKER: _JA.INDEPENDENT,
    JokerId.CRAFTY_JOKER: _JA.INDEPENDENT,
    JokerId.HALF_JOKER: _JA.INDEPENDENT,
    JokerId.FOUR_FINGERS: None,
    JokerId.BANNER: _JA.INDEPENDENT,
    JokerId.MYSTIC_SUMMIT: _JA.INDEPENDENT,
    JokerId.MISPRINT: _JA.INDEPENDENT,
    JokerId.RAISED_FIST: _JA.ON_HELD,
    JokerId.FIBONACCI: _JA.ON_SCORED,
    JokerId.SCARY_FACE: _JA.ON_SCORED,
    JokerId.PAREIDOLIA: None,
    JokerId.EVEN_STEVEN: _JA.ON_SCORED,
    JokerId.ODD_TODD: _JA.ON_SCORED,
    JokerId.SCHOLAR: _JA.ON_SCORED,
    JokerId.BLACKBOARD: _JA.INDEPENDENT,
    JokerId.SPLASH: None,
    JokerId.BLUE_JOKER: _JA.INDEPENDENT,
    JokerId.SHORTCUT: None,
    JokerId.BARON: _JA.ON_HELD,
    JokerId.PHOTOGRAPH: _JA.ON_SCORED,
    JokerId.ANCIENT_JOKER_CLUBS: _JA.ON_SCORED,
    JokerId.ANCIENT_JOKER_DIAMONDS: _JA.ON_SCORED,
    JokerId.ANCIENT_JOKER_HEARTS: _JA.ON_SCORED,
    JokerId.ANCIENT_JOKER_SPADES: _JA.ON_SCORED,
    JokerId.WALKIE_TALKIE: _JA.ON_SCORED,
    JokerId.SMILEY_FACE: _JA.ON_SCORED,
    JokerId.ACROBAT: _JA.INDEPENDENT,
    JokerId.SMEARED_JOKER: None,
    JokerId.BLOODSTONE: _JA.ON_SCORED,
    JokerId.ARROWHEAD: _JA.ON_SCORED,
    JokerId.ONYX_AGATE: _JA.ON_SCORED,
    JokerId.FLOWER_POT: _JA.INDEPENDENT,
    JokerId.BLUEPRINT: None,
    JokerId.SEEING_DOUBLE: _JA.INDEPENDENT,
    JokerId.THE_DUO: _JA.INDEPENDENT,
    JokerId.THE_TRIO: _JA.INDEPENDENT,
    JokerId.THE_FAMILY: _JA.INDEPENDENT,
    JokerId.THE_ORDER: _JA.INDEPENDENT,
    JokerId.THE_TRIBE: _JA.INDEPENDENT,
    JokerId.BRAINSTORM: None,
    JokerId.SHOOT_THE_MOON: _JA.ON_HELD,
    JokerId.TRIBOULET: _JA.ON_SCORED,
}

if set(JOKER_ACTIVATION.keys()) != set(JokerId) or len(JOKER_ACTIVATION) != NUM_JOKERS:
    raise AssertionError("JOKER_ACTIVATION must be dense: exactly one entry per JokerId")


JOKER_LABELS: dict[JokerId, str] = {
    JokerId.JOKER: "Joker",
    JokerId.GREEDY_JOKER: "Greedy Joker",
    JokerId.LUSTY_JOKER: "Lusty Joker",
    JokerId.WRATHFUL_JOKER: "Wrathful Joker",
    JokerId.GLUTTONOUS_JOKER: "Gluttonous Joker",
    JokerId.JOLLY_JOKER: "Jolly Joker",
    JokerId.ZANY_JOKER: "Zany Joker",
    JokerId.MAD_JOKER: "Mad Joker",
    JokerId.CRAZY_JOKER: "Crazy Joker",
    JokerId.DROLL_JOKER: "Droll Joker",
    JokerId.SLY_JOKER: "Sly Joker",
    JokerId.WILY_JOKER: "Wily Joker",
    JokerId.CLEVER_JOKER: "Clever Joker",
    JokerId.DEVIOUS_JOKER: "Devious Joker",
    JokerId.CRAFTY_JOKER: "Crafty Joker",
    JokerId.HALF_JOKER: "Half Joker",
    JokerId.FOUR_FINGERS: "Four Fingers",
    JokerId.BANNER: "Banner",
    JokerId.MYSTIC_SUMMIT: "Mystic Summit",
    JokerId.MISPRINT: "Misprint",
    JokerId.RAISED_FIST: "Raised Fist",
    JokerId.FIBONACCI: "Fibonacci",
    JokerId.SCARY_FACE: "Scary Face",
    JokerId.PAREIDOLIA: "Pareidolia",
    JokerId.EVEN_STEVEN: "Even Steven",
    JokerId.ODD_TODD: "Odd Todd",
    JokerId.SCHOLAR: "Scholar",
    JokerId.BLACKBOARD: "Blackboard",
    JokerId.SPLASH: "Splash",
    JokerId.BLUE_JOKER: "Blue Joker",
    JokerId.SHORTCUT: "Shortcut",
    JokerId.BARON: "Baron",
    JokerId.PHOTOGRAPH: "Photograph",
    JokerId.ANCIENT_JOKER_CLUBS: "Ancient Joker (Clubs)",
    JokerId.ANCIENT_JOKER_DIAMONDS: "Ancient Joker (Diamonds)",
    JokerId.ANCIENT_JOKER_HEARTS: "Ancient Joker (Hearts)",
    JokerId.ANCIENT_JOKER_SPADES: "Ancient Joker (Spades)",
    JokerId.WALKIE_TALKIE: "Walkie Talkie",
    JokerId.SMILEY_FACE: "Smiley Face",
    JokerId.ACROBAT: "Acrobat",
    JokerId.SMEARED_JOKER: "Smeared Joker",
    JokerId.BLOODSTONE: "Bloodstone",
    JokerId.ARROWHEAD: "Arrowhead",
    JokerId.ONYX_AGATE: "Onyx Agate",
    JokerId.FLOWER_POT: "Flower Pot",
    JokerId.BLUEPRINT: "Blueprint",
    JokerId.SEEING_DOUBLE: "Seeing Double",
    JokerId.THE_DUO: "The Duo",
    JokerId.THE_TRIO: "The Trio",
    JokerId.THE_FAMILY: "The Family",
    JokerId.THE_ORDER: "The Order",
    JokerId.THE_TRIBE: "The Tribe",
    JokerId.BRAINSTORM: "Brainstorm",
    JokerId.SHOOT_THE_MOON: "Shoot the Moon",
    JokerId.TRIBOULET: "Triboulet",
}
