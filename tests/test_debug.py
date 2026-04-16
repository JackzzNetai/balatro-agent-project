from defs import BossBlind, HandType, NO_BOSS_BLIND_ID
from debug import format_snapshot
from engine import Card, Joker
from utils import minimal_snapshot


def test_format_snapshot_contains_core_fields_and_hand_card():
    snap = minimal_snapshot(
        target_score=300,
        current_score=50,
        blind_id=NO_BOSS_BLIND_ID,
        hand=[Card(0, 0, 0), Card(51, 1, 2)],
        deck=[Card(3, 0, 0)],
        jokers=[Joker(5, 0)],
        play_remaining=4,
        discard_remaining=3,
        player_hand_size=5,
        hand_levels={int(HandType.HIGH_CARD): [10, 1]},
    )
    text = format_snapshot(snap)
    assert "target_score:      300" in text
    assert "current_score:     50" in text
    assert "non-boss blind" in text
    assert "--- Hand ---" in text
    assert "A♣" in text
    assert "K♠" in text
    assert "Bonus Card" in text
    assert "Holographic" in text
    assert "--- Deck ---" in text
    assert "1 card(s)" in text
    assert "--- Jokers ---" in text
    assert "Jolly Joker" in text
    assert "High Card" in text
    assert "chips=10" in text
    assert "mult=1" in text


def test_format_snapshot_boss_blind_label():
    snap = minimal_snapshot(blind_id=int(BossBlind.THE_HOOK))
    text = format_snapshot(snap)
    assert "The Hook" in text


def test_format_snapshot_deck_full_lists_all_cards():
    snap = minimal_snapshot(deck=[Card(0, 0, 0), Card(1, 0, 0)])
    text = format_snapshot(snap, deck="full")
    assert "[0] A♣" in text
    assert "[1] 2♣" in text
    assert "first " not in text
