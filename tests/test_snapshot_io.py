"""``env.snapshot_io`` JSON load/save (one object per file)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _tiny_snapshot():
    from defs import HandType, JokerId
    from engine import Card, GameSnapshot, Joker

    hl = {int(h): 2 for h in HandType}
    return GameSnapshot(
        100,
        0,
        -1,
        [Card(0, 0, 0), Card(1, 0, 0)],
        [Card(2, 0, 0)] * 12,
        [Joker(int(JokerId.JOKER), 0)],
        2,
        1,
        8,
        hl,
    )


def test_load_snapshot_roundtrip(tmp_path: Path):
    from env.snapshot_io import dict_to_snapshot, load_snapshot, save_snapshot, snapshot_to_dict

    snap = _tiny_snapshot()
    path = tmp_path / "one.json"
    save_snapshot(path, snap)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == snapshot_to_dict(snap)
    loaded = load_snapshot(path)
    assert loaded.target_score == snap.target_score
    assert loaded.hand_levels == snap.hand_levels


def test_load_snapshot_rejects_json_array(tmp_path: Path):
    from env.snapshot_io import load_snapshot

    path = tmp_path / "bad.json"
    path.write_text("[{}]", encoding="utf-8")
    with pytest.raises(TypeError, match="expected one JSON object"):
        load_snapshot(path)


def test_load_snapshot_pool_from_json_dir(tmp_path: Path):
    from env.snapshot_io import load_snapshot_pool_from_json_dir, save_snapshot

    save_snapshot(tmp_path / "a.json", _tiny_snapshot())
    save_snapshot(tmp_path / "b.json", _tiny_snapshot())
    pool = load_snapshot_pool_from_json_dir(tmp_path)
    assert len(pool) == 2


def test_snapshot_json_hand_levels_are_integer_levels(tmp_path: Path):
    from env.snapshot_io import load_snapshot, save_snapshot, snapshot_to_dict

    snap = _tiny_snapshot()
    d = snapshot_to_dict(snap)
    assert all(isinstance(v, int) for v in d["hand_levels"].values())
    path = tmp_path / "x.json"
    save_snapshot(path, snap)
    loaded = load_snapshot(path)
    assert loaded.hand_levels == snap.hand_levels


def test_dict_to_snapshot_level_only_hand_levels():
    from defs import HandType
    from env.snapshot_io import dict_to_snapshot

    d = {
        "target_score": 100,
        "current_score": 0,
        "blind_id": -1,
        "hand": [],
        "deck": [],
        "jokers": [],
        "play_remaining": 1,
        "discard_remaining": 0,
        "player_hand_size": 8,
        "hand_levels": {str(int(HandType.PAIR)): 2},
    }
    s = dict_to_snapshot(d)
    assert s.hand_levels[int(HandType.PAIR)] == 2
