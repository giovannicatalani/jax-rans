"""Audit bounded real campaign lanes and persist every Newton rejection.

Example: python scripts/diagnose_precision.py --precision fp32 --out /tmp/wing-fp32
The default is root section, three incidences, 40 steps; no production results
are overwritten. This checks stepping, not convergence of a complete campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

# Avoid competing with a production run's preallocated device-memory pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def timing_record(blocks, total_seconds):
    """Exclude first-use compilation, including a differently sized tail scan."""
    seen = set()
    samples = []
    for block in blocks:
        length = block["iterations"]
        if length in seen:
            samples.append(block["seconds"] / length)
        seen.add(length)
    return {"first_block_seconds": blocks[0]["seconds"],
            "total_solver_seconds": total_seconds,
            "steady_seconds_per_iteration": statistics.median(samples) if samples else None,
            "steady_sample_count": len(samples), "blocks": blocks}


def compare_reports(fp64, fp32):
    """Return a speedup only for the same completed, finite, moving workload."""
    runs = []
    for directory in (Path(fp64), Path(fp32)):
        config = json.loads((directory / "configuration.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        timing = json.loads((directory / "timing.json").read_text())
        if any(not s["finite_fields"] or s["exit_reason"] != "max_iters" for s in summary):
            raise ValueError(f"{directory}: failed or early-stopping cases invalidate the timing comparison")
        if any(b["accepted_steps"] != b["case_steps"] or b["changed_steps"] != b["case_steps"]
               for b in timing["blocks"]):
            raise ValueError(f"{directory}: rejected or frozen steps invalidate this fixed-work comparison")
        if not timing["steady_sample_count"] or not timing["steady_seconds_per_iteration"]:
            raise ValueError("at least two monitoring blocks are needed to exclude compilation")
        runs.append((config, summary, timing))
    a, b = runs
    def workload(config, summary):
        document = json.loads(json.dumps(config["campaign"]))
        document["general"].pop("precision", None)
        environment = {k: v for k, v in config["environment"].items() if k != "timestamp"}
        return (config["jobs"], config["numerical"], document, config["device"],
                environment, config["source_sha256"],
                [(s["section"], s["mach"], s["alpha_deg"], s["iterations"]) for s in summary])
    if workload(*a[:2]) != workload(*b[:2]):
        raise ValueError("workload, device, environment or solver source differs between runs")
    if a[0]["precision"] != "fp64" or b[0]["precision"] != "fp32":
        raise ValueError("expected an FP64 reference and an FP32 comparison")
    t64, t32 = a[2]["steady_seconds_per_iteration"], b[2]["steady_seconds_per_iteration"]
    if not all(math.isfinite(t) and t > 0 for t in (t64, t32)):
        raise ValueError("invalid timing samples")
    return {"fp64_ms_per_iteration": 1000*t64, "fp32_ms_per_iteration": 1000*t32,
            "steady_speedup": t64/t32,
            "fp64_first_block_seconds": a[2]["first_block_seconds"],
            "fp32_first_block_seconds": b[2]["first_block_seconds"],
            "fp64_total_solver_seconds": a[2]["total_solver_seconds"],
            "fp32_total_solver_seconds": b[2]["total_solver_seconds"],
            "validation_scope": "fixed-iteration speed; equal time to convergence is not established"}


def run_comparison(args):
    """Separate processes prevent one precision from retaining the other's GPU buffers."""
    for precision in ("fp64", "fp32"):
        command = [sys.executable, str(Path(__file__).resolve()), "--case", str(args.case),
                   "--precision", precision, "--backend", args.backend,
                   "--sections", *map(str, args.sections), "--cases", str(args.cases),
                   "--iters", str(args.iters), "--out", str(args.out / precision)]
        subprocess.run(command, check=True)
    result = compare_reports(args.out / "fp64", args.out / "fp32")
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2))
    print(f"Hors premier bloc : FP64 {result['fp64_ms_per_iteration']:.3f} ms/itération ; "
          f"FP32 {result['fp32_ms_per_iteration']:.3f} ms/itération ; "
          f"gain {result['steady_speedup']:.2f}x", flush=True)
    print("Mesure à nombre d’itérations fixé ; convergence équivalente non établie.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("examples/wing_sections"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--precision", choices=("fp32", "fp64"))
    mode.add_argument("--compare", action="store_true", help="time FP64 then FP32 in separate processes")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--sections", type=int, nargs="+", default=[0])
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--iters", type=int, help="default: 40 for audit, 100 for comparison")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.iters is None:
        args.iters = 100 if args.compare else 40
    if args.cases < 1 or args.iters < 1:
        parser.error("--cases and --iters must be positive")
    if args.compare:
        return run_comparison(args)

    import jax
    import numpy as np
    from jaxrans.campaign import _case, _write_solver_progress, load_campaign
    from jaxrans.campaign_batch import run_case_batch
    from jaxrans.geometry.icms import read_icms, write_hug
    from jaxrans.io.output import environment_record, git_revision

    campaign = load_campaign(args.case, (f"case.general.precision={args.precision}",
                                        f"numerical.convergence.max_iters={args.iters}",
                                        "numerical.convergence.forces_tol=null"))
    sections = read_icms(campaign.directory / campaign.sections["file"], campaign.sections["count"],
                         campaign.sections["units"], campaign.sections["header_lines"],
                         campaign.sections["plane_tolerance"])
    if any(s < 0 or s >= len(sections) for s in args.sections):
        parser.error("section index out of range")
    jobs = [(s, m, a) for m in range(len(campaign.machs)) for a in range(len(campaign.alphas))
            for s in args.sections][:args.cases]
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostic_path = args.out / "solver_progress.csv"
    diagnostic_path.write_text("")
    cases, grids, by_section = [], [], {}
    device = jax.devices(args.backend)[0]
    with jax.default_device(device):
        for s, m, a in jobs:
            hug = args.out.resolve() / f"section_{s:03d}.hug"
            if s not in by_section:
                write_hug(sections[s], hug)
            case = _case(campaign, hug, sections[s], campaign.machs[m], campaign.alphas[a])
            if s not in by_section:
                by_section[s] = case.grid()
            cases.append(case)
            grids.append(by_section[s])
        config = {"campaign": campaign.document, "numerical": campaign.numerical,
                  "jobs": jobs, "precision": args.precision, "device": str(device),
                  "environment": environment_record(), "git": git_revision(),
                  "validation_scope": "bounded stepping only; no full-convergence claim"}
        root = Path(__file__).resolve().parents[1]
        config["source_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src" / "jaxrans").rglob("*.py"))}
        (args.out / "configuration.json").write_text(json.dumps(config, indent=2))
        print(f"Audit {args.precision}: {len(cases)} cases, {args.iters} steps on {device}", flush=True)

        blocks = []

        def monitor(record):
            _write_solver_progress(diagnostic_path, record, jobs, campaign, 0)
            blocks.append({"seconds": record["block_seconds"], "iterations": record["block_iters"],
                           "case_steps": len(cases)*record["block_iters"],
                           "accepted_steps": int(record["newton"].accepted.sum()) if record["newton"] is not None
                           else int(record["changed_steps"].sum()),
                           "changed_steps": int(record["changed_steps"].sum())})
            labels = ["running" if str(r) == "max_iters" and record["iteration"] < args.iters
                      else str(r) for r in record["reasons"]]
            print(f"iteration {record['iteration']}: {labels}; "
                  f"changed steps {record['changed_steps'].tolist()}", flush=True)

        print("Premier bloc : compilation et exécution, exclu de la mesure du régime établi.", flush=True)
        results = run_case_batch(cases, grids, monitor=monitor)
    timing = timing_record(blocks, results[0].timing.total)
    (args.out / "timing.json").write_text(json.dumps(timing, indent=2))
    summary = [{"section": s, "mach": campaign.machs[m], "alpha_deg": campaign.alphas[a],
                "iterations": r.history.n_iters, "exit_reason": r.history.reason,
                "cl": r.forces.cl, "cd": r.forces.cd,
                "field_dtype": str(r.U.dtype),
                "finite_fields": bool(np.isfinite(r.U).all())}
               for (s, m, a), r in zip(jobs, results)]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    # Keep the exact GPU fields for CPU replay if a lane fails. Host copies
    # already exist in the results; this adds no device-memory allocation.
    np.savez_compressed(args.out / "states.npz", **{f"lane_{k:03d}": r.U for k, r in enumerate(results)})
    print(f"Diagnostics: {diagnostic_path}", flush=True)
    return int(any(r["exit_reason"] in ("stalled", "non_finite") for r in summary))


if __name__ == "__main__":
    raise SystemExit(main())
