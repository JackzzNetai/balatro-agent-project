"""Boss blind ids for the subset modeled in this gym.

``blind_id == NO_BOSS_BLIND_ID`` means no boss blind (e.g. Small / Big round).
Valid bosses use :class:`BossBlind` values ``0``..``6`` in a **custom order**
(grouped by effect—round / draw rules, then per-suit debuffs, then faces).
Names and effects match the wiki:
https://balatrowiki.org/w/Blinds_and_Antes

Prefer ``from defs import BossBlind, BOSS_BLIND_LABELS, NO_BOSS_BLIND_ID``.
"""

from __future__ import annotations

from enum import IntEnum

NO_BOSS_BLIND_ID = -1
"""Sentinel for ``GameSnapshot.blind_id`` when the current blind is not a boss."""


class BossBlind(IntEnum):
    """Dense ``0``..``6``: Hook and Serpent, four suit debuffs, then Plant."""

    THE_HOOK = 0
    THE_SERPENT = 1
    THE_CLUB = 2
    THE_WINDOW = 3
    THE_HEAD = 4
    THE_GOAD = 5
    THE_PLANT = 6


BOSS_BLIND_LABELS: dict[BossBlind, str] = {
    BossBlind.THE_HOOK: "The Hook",
    BossBlind.THE_SERPENT: "The Serpent",
    BossBlind.THE_CLUB: "The Club",
    BossBlind.THE_WINDOW: "The Window",
    BossBlind.THE_HEAD: "The Head",
    BossBlind.THE_GOAD: "The Goad",
    BossBlind.THE_PLANT: "The Plant",
}
