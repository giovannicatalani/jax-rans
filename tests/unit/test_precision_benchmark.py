"""Timing excludes cold compilation and rejects comparisons with failed work."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "precision_audit", Path(__file__).resolve().parents[2] / "scripts/diagnose_precision.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_timing_excludes_initial_and_partial_scan_compilation():
    blocks = [{"seconds": 100., "iterations": 20},
              {"seconds": .04, "iterations": 20},
              {"seconds": 99., "iterations": 10}]
    timing = audit.timing_record(blocks, 101.)
    assert timing["first_block_seconds"] == 100.
    assert timing["steady_seconds_per_iteration"] == pytest.approx(.002)
    assert timing["steady_sample_count"] == 1


@pytest.fixture
def reports(tmp_path):
    paths = [tmp_path / "fp64", tmp_path / "fp32"]
    for k, (path, precision, elapsed) in enumerate(zip(paths, ("fp64", "fp32"), (.1, .02))):
        path.mkdir()
        config = {"campaign": {"general": {"precision": precision}, "domain": {"cells": 70592}},
                  "jobs": [[0, 0, 0]], "numerical": {"iterations": 40}, "device": "cuda:0",
                  "environment": {"jax": "test", "timestamp": f"run-{k}"},
                  "source_sha256": {"solver": "same"}, "precision": precision}
        summary = [{"section": 0, "mach": .3, "alpha_deg": -2., "iterations": 40,
                    "finite_fields": True, "exit_reason": "max_iters"}]
        blocks = [{"seconds": 10., "iterations": 20, "case_steps": 20,
                   "accepted_steps": 20, "changed_steps": 20},
                  {"seconds": elapsed, "iterations": 20, "case_steps": 20,
                   "accepted_steps": 20, "changed_steps": 20}]
        for name, data in (("configuration.json", config), ("summary.json", summary),
                           ("timing.json", audit.timing_record(blocks, 12.))):
            (path / name).write_text(json.dumps(data))
    return paths


def test_comparison_reports_the_measured_ratio_not_theoretical_flops(reports):
    result = audit.compare_reports(*reports)
    assert result["steady_speedup"] == pytest.approx(5.)
    assert result["fp64_ms_per_iteration"] == pytest.approx(5.)
    assert result["fp32_ms_per_iteration"] == pytest.approx(1.)


@pytest.mark.parametrize("file,mutation", [
    ("summary.json", lambda d: d[0].update(exit_reason="stalled")),
    ("summary.json", lambda d: d[0].update(finite_fields=False)),
    ("summary.json", lambda d: d[0].update(iterations=20)),
    ("configuration.json", lambda d: d.update(device="cpu:0")),
    ("configuration.json", lambda d: d["source_sha256"].update(solver="different")),
    ("timing.json", lambda d: d["blocks"][1].update(accepted_steps=0)),
])
def test_invalid_work_cannot_be_reported_as_a_speedup(reports, file, mutation):
    path = reports[1] / file
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit.compare_reports(*reports)
