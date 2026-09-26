import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from finding_level_gaze_targets.data.reflacx import _seconds, find_records
from finding_level_gaze_targets.linking.rules import (
    apply_lookback,
    assert_cached_lookback,
)


ROOT = Path(__file__).resolve().parents[1]


def test_configuration_registry_readme_and_entry_points_agree():
    config = json.loads((ROOT / "configs/study.json").read_text(encoding="utf-8"))
    registry = json.loads(
        (ROOT / config["result_registry"]).read_text(encoding="utf-8")
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    analyses = runpy.run_path(str(ROOT / "scripts/run_analysis.py"))["ANALYSES"]

    assert config["title"] == registry["associated_publication"]["title"]
    assert registry["project"] == {
        "name": "Finding-Level Gaze Targets",
        "python_package": "finding_level_gaze_targets",
        "repository": "https://github.com/TeamBeaverSudal/cxr-fixation-attribution",
    }
    assert set(config["analysis_entry_points"].values()) == set(analyses)
    assert "987 mention-linked finding instances from 398" in readme
    assert "0.7893" in readme and "0.3549" in readme
    assert "0.8245" in readme and "0.3584" in readme

    for module in analyses.values():
        relative = Path("src", *module.split(".")).with_suffix(".py")
        assert (ROOT / relative).is_file(), module


def test_current_source_tree_excludes_superseded_driver_names():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src").rglob("*.py"))
    )
    for obsolete in (
        "b1_scan_heat",
        "word_dirs",
        "reflacx_io.py",
        "_stage3_selfcheck",
    ):
        assert obsolete not in source


def test_reflacx_record_index_and_time_units(tmp_path):
    first = tmp_path / "version" / "record-a"
    first.mkdir(parents=True)
    for filename in (
        "fixations.csv",
        "anomaly_location_ellipses.csv",
        "timestamps_transcription.csv",
    ):
        (first / filename).write_text("header\n", encoding="utf-8")

    records = find_records(tmp_path)
    assert set(records) == {"record-a"}
    assert set(records["record-a"]) == {
        "fixations.csv",
        "anomaly_location_ellipses.csv",
        "timestamps_transcription.csv",
    }
    assert np.array_equal(_seconds([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.array_equal(_seconds([0.0, 5000.0]), np.array([0.0, 5.0]))


def test_reflacx_record_index_rejects_duplicate_versions(tmp_path):
    for version in ("v1", "v2"):
        record = tmp_path / version / "record-a"
        record.mkdir(parents=True)
        (record / "fixations.csv").write_text("header\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ambiguous fixations.csv"):
        find_records(tmp_path)


def test_extended_lookback_is_reconstructed_from_sentence_bounds():
    base = [(0.0, 2.0, 3.0), (3.0, 5.0, 6.0)]
    cached = [(0.5, 2.0, 3.0), (3.5, 5.0, 6.0)]
    assert apply_lookback(base, 1.5) == cached
    assert apply_lookback(base, 3.0) == [(0.0, 2.0, 3.0), (3.0, 5.0, 6.0)]
    assert_cached_lookback(cached, base)
    with pytest.raises(RuntimeError, match="reconstruction mismatch"):
        assert_cached_lookback([(0.0, 2.0, 3.0)], base)
