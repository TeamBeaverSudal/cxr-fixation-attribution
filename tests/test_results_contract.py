import json
from pathlib import Path

from finding_level_gaze_targets.reporting.registry import validate_registry


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "study-results.json"


def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_registry_is_valid():
    assert validate_registry(load_results()) == []


def test_primary_inference_matches_reported_study():
    data = load_results()
    comparison = data["primary_inference"]["learned_minus_structured_3_0s"]
    assert comparison["pointing"] == {
        "difference": 0.0353,
        "ci95": [0.0067, 0.0634],
        "patient_signed_rank_p": 0.0766815,
    }
    assert comparison["iou"] == {
        "difference": 0.0035,
        "ci95": [-0.0034, 0.0102],
        "patient_signed_rank_p": 0.841748,
    }


def test_all_partitions_use_five_optimizer_seeds():
    rows = load_results()["patient_partitions"]
    assert [row["partition"] for row in rows] == [0, 1, 2, 3, 4]
    assert {row["optimizer_seeds"] for row in rows} == {5}


def test_training_fraction_matrix_is_complete():
    table = load_results()["table_4"]
    assert table["completed_tasks"] == 25
    assert table["patient_subsample_chains"] == 5
    assert table["optimizer_seeds_per_chain"] == 5
    assert [row["fraction"] for row in table["rows"]] == [0.1, 0.25, 0.5, 1.0]
