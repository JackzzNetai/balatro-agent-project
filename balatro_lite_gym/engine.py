from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Card:
    """Playing card. ``card_id`` in ``0..51`` with ``card_id = suit * 13 + rank``:

    ``rank = card_id % NUM_RANKS`` (0=Ace … 12=King), ``suit = card_id // NUM_RANKS`` (0..3);
    see :mod:`defs` and :mod:`util` for enums and ``card_id`` helpers.
    ``enhancement`` / ``edition`` use ids from ``defs.CardEnhancement`` and
    ``defs.Edition`` (defaults: ``enhancement == 0`` = Plain / ``NONE``, ``edition == 0`` = Base).
    """

    card_id: int
    enhancement: int
    edition: int


@dataclass
class Joker:
    """``edition`` uses ``defs.Edition`` ids (default ``0`` = Base)."""

    id: int
    edition: int


@dataclass
class GameSnapshot:
    """Environment snapshot.

    ``hand_levels``: keys are ``HandType`` ints (see ``defs.HandType``);
    each value is ``[chips, mult]`` — direct chip and **additive** mult from the
    poker hand line for that type (not planet level indices). Scoring a play
    requires a key for the classified hand (see ``scoring.score_play``).

    ``blind_id``: ``defs.NO_BOSS_BLIND_ID`` (``-1``) when the round is not a boss
    blind (e.g. Small / Big); otherwise a ``defs.BossBlind`` value (``0``..``6``).
    """

    target_score: int
    current_score: int
    blind_id: int
    hand: List[Card]
    deck: List[Card]
    jokers: List[Joker]
    play_remaining: int
    discard_remaining: int
    player_hand_size: int
    hand_levels: Dict[int, List[int]]
