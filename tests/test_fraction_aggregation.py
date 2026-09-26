from pathlib import Path

import pytest

from finding_level_gaze_targets.experiments.summarize_consistent_runs import (
    fraction_summary,
)


def populate(root: Path) -> None:
    for subset in range(5):
        for model in range(5):
            run_dir = root / "training-fraction" / f"subset-{subset}" / f"seed-{model}"
            run_dir.mkdir(parents=True)
            (run_dir / "COMPLETE").touch()
            fractions = [0.10, 0.25, 0.50] + ([1.00] if subset == 0 else [])
            lines = []
            for fraction in fractions:
                if model == 0:
                    lines.append(
                        "FRACTION_RESULT method=STRUCTURED "
                        f"fraction={fraction:.2f} subset_seed={subset} n=987 "
                        f"pg={0.70 + fraction / 10:.4f} "
                        f"iou={0.30 + fraction / 100:.4f} lookback=3.0"
                    )
                lines.append(
                    "FRACTION_RESULT method=LEARNED "
                    f"fraction={fraction:.2f} subset_seed={subset} n=987 "
                    f"pg={0.72 + fraction / 10 + model / 1000:.4f} "
                    f"iou={0.31 + fraction / 100 + model / 10000:.4f}"
                )
            (run_dir / "run.log").write_text("\n".join(lines) + "\n")


def test_crossed_fraction_aggregation(tmp_path):
    populate(tmp_path)
    result = fraction_summary(tmp_path)
    assert result["0.10"]["learned"]["pg"]["n"] == 5
    assert result["1.00"]["learned"]["pg"]["n"] == 1
    assert result["0.50"]["structured"][
        "selected_lookback_seconds"
    ]["by_patient_subset"] == [3.0] * 5


def test_missing_completion_marker_fails_closed(tmp_path):
    populate(tmp_path)
    marker = tmp_path / "training-fraction" / "subset-4" / "seed-4" / "COMPLETE"
    marker.unlink()
    with pytest.raises(RuntimeError, match="run is not complete"):
        fraction_summary(tmp_path)
