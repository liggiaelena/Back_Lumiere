import numpy as np

from dev.app.pipeline import _confirm_melasma_candidate


def _candidate():
    return {
        "melasma": {
            "mask": np.ones((4, 4), dtype=np.uint8),
            "threshold": 0.55,
            "area_percent": 8.0,
            "zones": ["forehead", "left_cheek", "right_cheek"],
        }
    }


def test_soft_melasma_requires_two_supporting_spot_regions():
    condition_map = {"melasma": {"detected": False}}
    report = {
        "imperfeicoes": [
            {"tipo": "mancha", "regiao": "testa"},
            {"tipo": "mancha", "regiao": "bochecha_e"},
        ]
    }

    mask = _confirm_melasma_candidate(report, condition_map, _candidate())

    assert mask is not None
    assert condition_map["melasma"]["detected"] is True
    assert condition_map["melasma"]["suspected"] is True


def test_single_spot_does_not_confirm_soft_melasma():
    condition_map = {"melasma": {"detected": False}}
    report = {"imperfeicoes": [{"tipo": "mancha", "regiao": "testa"}]}

    mask = _confirm_melasma_candidate(report, condition_map, _candidate())

    assert mask is None
    assert condition_map["melasma"]["detected"] is False
