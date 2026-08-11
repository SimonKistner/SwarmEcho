import csv

import numpy as np
import pytest

from swarmecho.training.artifacts import (
    evaluation_stage_from_rates,
    load_eval_info_csv,
    save_eval_info_csv,
)


def test_robust_eval_csv_stores_cumulative_rates_and_uses_unanimous_stage(tmp_path):
    path = save_eval_info_csv(
        tmp_path / "eval.csv",
        target_positions=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        base_positions=np.zeros((2, 3)),
        stage_rates={
            "chain_success": np.asarray([0.8, 1.0]),
            "found_and_delivered": np.asarray([0.8, 1.0]),
            "visually_found": np.asarray([1.0, 1.0]),
        },
    )

    with path.open(newline="") as stream:
        assert next(csv.reader(stream)) == [
            "x",
            "y",
            "z",
            "visually_found_rate",
            "found_and_delivered_rate",
            "chain_success_rate",
            "distance_to_base",
        ]

    records = load_eval_info_csv(path)
    assert records["stages"].tolist() == ["visually_found", "chain_success"]
    np.testing.assert_allclose(records["chain_success_rate"], [0.8, 1.0])
    np.testing.assert_allclose(records["found_and_delivered_rate"], [0.8, 1.0])
    np.testing.assert_allclose(records["visually_found_rate"], [1.0, 1.0])


def test_confidence_selects_highest_cumulative_stage_that_meets_threshold():
    rates = {
        "chain_success_rate": 0.6,
        "found_and_delivered_rate": 0.8,
        "visually_found_rate": 1.0,
    }

    assert evaluation_stage_from_rates(**rates, confidence=1.0) == "visually_found"
    assert (
        evaluation_stage_from_rates(**rates, confidence=0.8)
        == "found_and_delivered"
    )
    assert evaluation_stage_from_rates(**rates, confidence=0.5) == "chain_success"


def test_robust_eval_csv_rejects_non_cumulative_rates(tmp_path):
    with pytest.raises(ValueError, match="chain <= delivered <= visually_found"):
        save_eval_info_csv(
            tmp_path / "bad.csv",
            target_positions=np.asarray([[1.0, 2.0, 3.0]]),
            base_positions=np.zeros((1, 3)),
            stage_rates={
                "chain_success": np.asarray([1.0]),
                "found_and_delivered": np.asarray([0.8]),
                "visually_found": np.asarray([1.0]),
            },
        )
