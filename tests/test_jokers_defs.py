"""``defs.jokers`` — dense ids, wiki ordering, Ancient block placement."""

from defs import (
    JOKER_ACTIVATION,
    JOKER_ID_HIGH,
    JOKER_LABELS,
    JokerActivation,
    JokerId,
    NUM_JOKERS,
)


def test_joker_id_count_and_range():
    assert NUM_JOKERS == 55
    assert JOKER_ID_HIGH == 54
    assert len(JokerId) == NUM_JOKERS
    assert min(JokerId) == 0
    assert max(JokerId) == JOKER_ID_HIGH
    values = {int(m) for m in JokerId}
    assert len(values) == NUM_JOKERS
    assert values == set(range(NUM_JOKERS))


def test_ancient_block_after_photograph_before_walkie():
    assert JokerId.PHOTOGRAPH == 32
    assert JokerId.ANCIENT_JOKER_CLUBS == 33
    assert JokerId.ANCIENT_JOKER_DIAMONDS == 34
    assert JokerId.ANCIENT_JOKER_HEARTS == 35
    assert JokerId.ANCIENT_JOKER_SPADES == 36
    assert JokerId.WALKIE_TALKIE == 37


def test_joker_labels_cover_all():
    assert len(JOKER_LABELS) == NUM_JOKERS
    for j in JokerId:
        assert j in JOKER_LABELS
        assert isinstance(JOKER_LABELS[j], str) and JOKER_LABELS[j]


def test_joker_activation_matches_effect_handlers_keys():
    """Every non-``None`` activation must have a stub in ``joker_effects``."""
    import joker_effects

    for jid in JokerId:
        act = JOKER_ACTIVATION[jid]
        if act is None:
            assert jid not in joker_effects.EFFECT_HANDLERS
        else:
            assert act in (
                JokerActivation.ON_SCORED,
                JokerActivation.ON_HELD,
                JokerActivation.INDEPENDENT,
            )
            assert jid in joker_effects.EFFECT_HANDLERS
