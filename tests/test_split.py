import json

import pytest

from maint_copilot.data import SPLIT_FILE, load_split


def test_committed_split_loads_and_sealed_hash_matches():
    dev, sealed = load_split("dev"), load_split("sealed")
    assert len(dev) == 113 and len(sealed) == 170
    assert not set(dev) & set(sealed)


def test_tampered_sealed_list_is_rejected(tmp_path):
    split = json.loads(SPLIT_FILE.read_text())
    split["sealed"] = split["sealed"][:-1]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    assert len(load_split("dev", path)) == 113
    with pytest.raises(ValueError, match="SHA-256"):
        load_split("sealed", path)


def test_unknown_split_name():
    with pytest.raises(ValueError):
        load_split("test")
