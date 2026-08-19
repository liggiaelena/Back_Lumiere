import numpy as np

from app import vitiligo_gate


def test_gate_suppresses_vitiligo_below_threshold(monkeypatch):
    monkeypatch.setattr(vitiligo_gate, "predict_vitiligo_probability", lambda _: 0.2)
    monkeypatch.setattr(vitiligo_gate, "_get_bundle", lambda: {"threshold": 0.35})
    mask = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    conditions = {
        "vitiligo": {"detected": True, "area_percent": 50.0, "zones": ["left_cheek"]}
    }

    vitiligo_gate.apply_vitiligo_gate(np.zeros((2, 2, 3)), mask, conditions)

    assert not conditions["vitiligo"]["detected"]
    assert conditions["vitiligo"]["suppressed_by_classifier"]
    assert not np.any(mask == 1)
    assert mask[1, 1] == 2


def test_gate_preserves_vitiligo_above_threshold(monkeypatch):
    monkeypatch.setattr(vitiligo_gate, "predict_vitiligo_probability", lambda _: 0.8)
    monkeypatch.setattr(vitiligo_gate, "_get_bundle", lambda: {"threshold": 0.35})
    mask = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    conditions = {
        "vitiligo": {"detected": True, "area_percent": 50.0, "zones": ["left_cheek"]}
    }

    vitiligo_gate.apply_vitiligo_gate(np.zeros((2, 2, 3)), mask, conditions)

    assert conditions["vitiligo"]["detected"]
    assert conditions["vitiligo"]["classifier_gate"]["passed"]
    assert np.sum(mask == 1) == 2


def test_missing_gate_fails_open(monkeypatch):
    monkeypatch.setattr(vitiligo_gate, "predict_vitiligo_probability", lambda _: None)
    mask = np.array([[0, 1]], dtype=np.uint8)
    conditions = {"vitiligo": {"detected": True}}

    vitiligo_gate.apply_vitiligo_gate(np.zeros((1, 2, 3)), mask, conditions)

    assert conditions["vitiligo"]["detected"]
    assert conditions["vitiligo"]["classifier_gate"] == {"available": False}
    assert mask[0, 1] == 1
