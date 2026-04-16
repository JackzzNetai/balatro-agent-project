from defs import HAND_TYPE_COUNT, HAND_TYPE_LABELS, HandType


def test_hand_type_count_matches_environment_contract():
    assert HAND_TYPE_COUNT == 12
    assert len(HandType) == 12


def test_hand_type_ids_are_contiguous_zero_through_eleven():
    values = [int(h) for h in HandType]
    assert values == list(range(12))


def test_labels_cover_all_hand_types():
    assert set(HAND_TYPE_LABELS.keys()) == set(HandType)


def test_straight_flush_is_eight_includes_royal_by_convention():
    """Royal flush is not a separate id; same as Straight Flush (wiki)."""
    assert HandType.STRAIGHT_FLUSH == 8
