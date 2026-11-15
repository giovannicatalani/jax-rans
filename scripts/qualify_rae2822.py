#!/usr/bin/env python3
"""Plan or run independent RAE2822 witness comparisons (no validation verdict).

See docs/rae2822-robustness.md. No solve is performed with --dry-run or
--manifest-only. A saved field supplies physical U only, never solver state.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jaxrans import load_case, run
from jaxrans.io.output import git_revision


ROOT = Path(__file__).resolve().parents[1]
WITNESSES = {
    "floor": {"mach": 0.745, "alpha_deg": -1.5},
    "sa": {"mach": 0.70, "alpha_deg": 4.0},
    "oscillation": {"mach": 0.76, "alpha_deg": -3.0},
    "lowmach": {"mach": 0.10, "alpha_deg": -2.5},
    "control": {"mach": 0.78, "alpha_deg": -3.0},
}
# Each patch is applied to the supplied baseline, never to the previous job.
# Several knobs can describe one factor (CFL schedule or GMRES capacity).
VARIANTS = {
    "baseline": {},
    "cautious_cfl": {"implicit.cfl": 2.0, "implicit.cfl_max": 50.0,
                     "implicit.cfl_growth": 1.05},
    "nonlinear_growth_guard": {"implicit.nonlinear_growth_limit": 1.5},
    "equation_growth_guard": {"implicit.nonlinear_growth_limit": 1.5,
                              "implicit.nonlinear_merit": "equations"},
    "van_albada": {"convective_fluxes.limiter": "van_albada"},
    "minmod": {"convective_fluxes.limiter": "minmod"},
    "gmres_capacity": {"implicit.krylov_subspace": 30, "implicit.krylov_restarts": 4},
    "update_fraction": {"implicit.max_relative_update": 0.1},
}
CRITERIA = ("tol", "forces_tol", "equation_tol", "equation_abs_tol", "equation_abs_tols", "max_iters")


def jsonable(value):
    """Strict JSON: absent/nonfinite measurements are null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dumps(value):
    return json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {"path": str(path), "sha256": checksum.hexdigest(), "bytes": path.stat().st_size}


def array_record(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", type=Path, required=True, help="directory with case.json/numerical.json")
    cli.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    cli.add_argument("--witness", nargs="+", choices=[*WITNESSES, "all"], default=["floor"])
    cli.add_argument("--variant", nargs="+", choices=[*VARIANTS, "all"], default=["baseline"])
    cli.add_argument("--precision", nargs="+", choices=["fp32", "fp64", "mixed", "mixt"], default=["fp32"])
    cli.add_argument("--max-iters", type=int, default=200, help="per-job ceiling (default: 200)")
    cli.add_argument("--initial", type=Path, help="Result.write *_fields.npz on the exact mesh")
    cli.add_argument("--set", dest="overrides", action="append", default=[], metavar="FILE.SECTION.KEY=VALUE",
                     help="common baseline override; repeatable, e.g. case.freestream.sweep_deg=null")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print manifest; write nothing")
    mode.add_argument("--manifest-only", action="store_true", help="write manifest; do not solve")
    return cli


def selections(given, catalog):
    return list(catalog) if "all" in given else list(dict.fromkeys(given))


def resolved_config(case):
    return {"flow": asdict(case.flow), "numerics": asdict(case.numerics), "implicit": asdict(case.implicit),
            "grid_spec": asdict(case.grid_spec), "conditions": case.conditions,
            "block_iters": case.block_iters, "sequencing": case.sequencing,
            "rk_scheme": case.rk_scheme}


def input_records(case):
    records = [file_record(case.directory / "case.json")]
    numerical = case.directory / "numerical.json"
    if numerical.is_file():
        records.append(file_record(numerical))
    spec = case.grid_spec
    for name in [spec.file, spec.airfoil, *(e.get("file") for e in spec.elements)]:
        if name is None or (name == spec.airfoil and str(name).lower().startswith("naca")):
            continue
        record = file_record(spec.resolve(case.directory, name))
        if record not in records:
            records.append(record)
    return records


def load_initial(path, grid, case):
    """Validate exact ordered nodes and physical block fields, then cast U.

    No interpolation, variable insertion, ghost import, or solver-state import.
    SA-neg is allowed: nu~ is not clipped or required to be positive.
    """
    nvar = grid.components + 2 + int(case.flow.turbulence is not None)
    blocks, sources, nodes_record = [], [], []
    dtype = case.numerics.dtype
    with np.load(path, allow_pickle=False) as saved:
        if "n_blocks" not in saved:
            raise ValueError("initial fields must contain n_blocks")
        count = saved["n_blocks"]
        if count.shape != () or count.dtype.kind not in "iu" or int(count) != len(grid.mesh):
            raise ValueError("initial n_blocks does not match the exact mesh")
        expected_keys = {"n_blocks"}
        for k, block in enumerate(grid.mesh):
            nodes = np.asarray(block.nodes)
            nodes_record.append(array_record(nodes))
            for axis, coordinates in zip("xyz", nodes):
                key = f"{axis}_node_{k}"
                expected_keys.add(key)
                if key not in saved or not np.array_equal(saved[key], coordinates):
                    raise ValueError(f"initial {key} does not match the exact mesh coordinates")
            key = f"U_{k}"
            expected_keys.add(key)
            if key not in saved:
                raise ValueError(f"initial missing {key}")
            source = saved[key]
            shape = (nvar, *(n - 1 for n in nodes.shape[1:]))
            if source.shape != shape or source.dtype.kind != "f":
                raise ValueError(f"initial {key} needs floating physical U with shape {shape}")
            sources.append(array_record(source))
            with np.errstate(over="ignore", invalid="ignore"):
                U = source.astype(dtype)
            if not np.isfinite(U).all() or np.any(U[0] <= 0):
                raise ValueError(f"initial {key} is nonfinite or has nonpositive density after cast")
            # Evaluate pressure in double to avoid overflow in the validation itself.
            rho = U[0].astype(np.float64)
            momentum = U[1:1 + grid.components].astype(np.float64)
            energy = U[1 + grid.components].astype(np.float64)
            internal = energy - np.sum((momentum / rho) * momentum, axis=0) / 2
            if not np.isfinite(internal).all() or np.any(internal <= 0):
                raise ValueError(f"initial {key} has nonpositive pressure after cast")
            blocks.append(U)
        if set(saved.files) != expected_keys:
            raise ValueError("initial must contain only Result.write physical fields and mesh nodes")
    packed = grid.pack(blocks)
    if np.dtype(packed.dtype) != dtype:
        raise ValueError("initial packing did not retain requested field precision")
    return packed, {"physical_U_sha256": digest(sources), "mesh_sha256": digest(nodes_record),
                    "source_blocks": sources, "field_dtype": str(dtype),
                    "restart": "physical_U_only; solver controls and histories reset"}


def build_plan(args):
    if args.max_iters < 1:
        raise ValueError("--max-iters must be positive")
    case_path = args.case.resolve(strict=True)
    common = ["numerical.convergence.forces_tol=0.0001",
              "numerical.convergence.grid_sequencing=0", *args.overrides]
    base = load_case(case_path, tuple(common))
    if base.numerics.solver != "implicit" or base.flow.turbulence is None:
        raise ValueError("witness comparison requires an implicit turbulent SA case")
    if base.grid_spec.grid not in ("c_grid", "o_grid", "file"):
        raise ValueError("witness comparison requires a single-airfoil structured mesh")
    if base.sequencing and not args.initial:
        raise ValueError("grid sequencing would change initial fields between variants; use --initial or sequencing=0")
    if args.max_iters % base.block_iters:
        raise ValueError("--max-iters must be a multiple of convergence.block_iters; "
                         "use --set numerical.convergence.block_iters=1 for an arbitrary budget")
    initial = ({"kind": "saved_fields", **file_record(args.initial)} if args.initial else
               {"kind": "freestream", "continuation_reproduction": False})
    source = {str(p.relative_to(ROOT)): file_record(p)["sha256"]
              for p in sorted((ROOT / "src/jaxrans").rglob("*.py"))}
    source[str(Path(__file__).resolve().relative_to(ROOT))] = file_record(__file__)["sha256"]
    inputs = input_records(base)
    jobs, cases, initial_validations = [], {}, {}
    witnesses = selections(args.witness, WITNESSES)
    variants = selections(args.variant, VARIANTS)
    precisions = list(dict.fromkeys(args.precision))
    for witness, variant, precision in itertools.product(witnesses, variants, precisions):
        conditions = WITNESSES[witness]
        baseline_overrides = [*common, *(f"case.freestream.{k}={v}" for k, v in conditions.items()),
                              f"case.general.precision={precision}",
                              f"numerical.convergence.max_iters={args.max_iters}",
                              "numerical.convergence.diagnostics=true"]
        overrides = [*baseline_overrides,
                     *(f"numerical.{k}={json.dumps(v)}" for k, v in VARIANTS[variant].items())]
        case = load_case(case_path, tuple(overrides))
        config = resolved_config(case)
        criteria = {k: v for k, v in asdict(case.numerics).items() if k in CRITERIA}
        criteria.update(block_iters=case.block_iters, sequencing=0 if args.initial else case.sequencing)
        start = dict(initial)
        if args.initial:
            mesh_key = digest([asdict(case.grid_spec), case.conditions, case.flow.sweep_deg,
                               case.flow.turbulence, case.numerics.n_ghost, precision])
            if mesh_key not in initial_validations:
                _, validated = load_initial(args.initial, case.grid(), case)
                initial_validations[mesh_key] = validated
            start.update(initial_validations[mesh_key])
        # Group by shared baseline, not by the variant's altered discretization.
        baseline = load_case(case_path, tuple(baseline_overrides))
        shared = resolved_config(baseline)
        shared["numerics"].pop("precision")
        identity = {"baseline": shared, "criteria": criteria, "initial": initial,
                    "inputs": inputs, "source": source}
        job_id = f"{witness}__{variant}__{precision}"
        job = {"id": job_id, "witness": witness, "variant": variant, "precision": precision,
               "directory": str(args.out.resolve() / job_id), "overrides": overrides,
               "factor_patch": VARIANTS[variant], "config": config, "raw": case.raw,
               "config_sha256": digest(config), "initial": start, "criteria": criteria,
               "comparison_group": digest(identity)}
        if args.initial:
            job["effective_sequencing"] = 0
        jobs.append(job)
        cases[job_id] = case
    manifest = {"schema_version": 1, "case": str(case_path), "out": str(args.out.resolve()),
                "git": git_revision(ROOT), "source_sha256": source, "inputs": inputs,
                "initial": initial, "jobs": jobs, "job_count": len(jobs),
                "interpretation": "Independent restarts; no automatic aerodynamic validation thresholds"}
    return manifest, cases


def result_record(result, job, elapsed):
    convergence = result.convergence_history[-1] if result.convergence_history else {}
    blockers = convergence.get("blockers")
    if blockers is None:
        if not convergence:
            blockers = ["convergence_diagnostics_unavailable"]
        else:
            blockers = list(convergence.get("blocking_equations", []))
            if convergence.get("forces_settled") is False:
                blockers.append("forces")
    return {"id": job["id"], "witness": job["witness"], "variant": job["variant"],
            "precision": job["precision"], "comparison_group": job["comparison_group"],
            "initial": job["initial"], "criteria": job["criteria"],
            "config_sha256": job["config_sha256"], "directory": job["directory"],
            "seconds": result.seconds, "run_wall_seconds": elapsed,
            "timing": result.timing.record(), "CL": result.forces.cl,
            "CD": result.forces.cd, "CM": result.forces.cm,
            "iterations": result.history.n_iters, "exit_reason": result.history.reason,
            "blockers": blockers,
            "convergence": convergence, "newton_blocks": result.history.block_diagnostics,
            "field_dtype": str(result.U.dtype), "status": "completed"}


def compare(reference, candidate, surfaces):
    """Only compare one variant or precision factor, with identical provenance."""
    if reference["comparison_group"] != candidate["comparison_group"]:
        raise ValueError("cannot compare different initial fields, criteria, or baseline inputs")
    different = [k for k in ("variant", "precision") if reference[k] != candidate[k]]
    if len(different) != 1:
        raise ValueError("comparison must change exactly one variant or precision factor")
    a, b = (surfaces[r["id"]] for r in (reference, candidate))
    coordinates = {k for k in a if k not in ("cp", "cf", "cf_z")}
    if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in coordinates):
        raise ValueError("surface coordinates/order differ; no interpolation is performed")
    distributions = {}
    for key in ("cp", "cf"):
        if key not in a:
            raise ValueError(f"Result.surface missing {key}")
        delta = np.asarray(b[key], dtype=np.float64) - np.asarray(a[key], dtype=np.float64)
        distributions[key] = {"linf": float(np.max(np.abs(delta))),
                              "rms": float(np.sqrt(np.mean(delta**2)))}
    return {"reference": reference["id"], "candidate": candidate["id"], "factor": different[0],
            "comparison_group": reference["comparison_group"],
            "exit_reasons": [reference["exit_reason"], candidate["exit_reason"]],
            "delta": {k: candidate[k] - reference[k] for k in ("CL", "CD", "CM", "seconds")},
            "surface_delta": distributions,
            "interpretation": "Measured differences, including unconverged runs; no pass/fail threshold"}


def execute(manifest, cases):
    records, surfaces, comparisons = [], {}, []
    out = Path(manifest["out"])
    failed = False
    for job in manifest["jobs"]:
        directory = Path(job["directory"])
        directory.mkdir()
        (directory / "job.json").write_text(dumps(job))
        started = time.perf_counter()
        try:
            case = cases[job["id"]]
            grid = case.grid()
            # Guard against replacing a file after preflight, including between jobs.
            initial = None
            if job["initial"]["kind"] == "saved_fields":
                path = job["initial"]["path"]
                if file_record(path)["sha256"] != job["initial"]["sha256"]:
                    raise ValueError("initial file changed since planning")
                initial, validated = load_initial(path, grid, case)
                if validated != {k: job["initial"][k] for k in validated}:
                    raise ValueError("initial field/mesh changed since planning")
            result = run(case, grid=grid, initial=initial)
            elapsed = time.perf_counter() - started
            result.write(directory, extra={"qualification": job})
            wall = result.surface()
            if "cp" not in wall or "cf" not in wall:
                raise ValueError("Result.write must export Cp and Cf for these viscous witnesses")
            surfaces[job["id"]] = wall
            record = result_record(result, job, elapsed)
            record["mesh_sha256"] = digest([array_record(b.nodes) for b in grid.mesh])
            # Exact physical mesh identity is required in addition to the plan identity.
            record["comparison_group"] = digest([job["comparison_group"], record["mesh_sha256"]])
            failed |= result.history.reason not in ("converged", "settled")
        except Exception as error:
            failed = True
            record = {"id": job["id"], "status": "error", "exit_reason": "error",
                      "seconds": None, "run_wall_seconds": time.perf_counter() - started,
                      "CL": None, "CD": None, "CM": None, "blockers": [str(error)],
                      "error_type": type(error).__name__, "initial": job["initial"],
                      "config_sha256": job["config_sha256"], "criteria": job["criteria"]}
        records.append(record)
        (directory / "summary.json").write_text(dumps(record))
        completed = [r for r in records if r["status"] == "completed"]
        comparisons = []
        for a, b in itertools.combinations(completed, 2):
            if a["comparison_group"] != b["comparison_group"]:
                continue
            same_precision = a["precision"] == b["precision"]
            if same_precision and "baseline" in (a["variant"], b["variant"]):
                a, b = (a, b) if a["variant"] == "baseline" else (b, a)
            elif a["variant"] != b["variant"] or same_precision:
                continue
            try:
                comparisons.append(compare(a, b, surfaces))
            except ValueError as error:
                failed = True
                comparisons.append({"reference": a["id"], "candidate": b["id"],
                                    "not_comparable": str(error)})
        (out / "summary.json").write_text(dumps({"runs": records, "comparisons": comparisons,
                                                "manifest_sha256": digest(manifest)}))
        print(f"{job['id']}: {record['exit_reason']}", flush=True)
    return 1 if failed else 0


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        manifest, cases = build_plan(args)
        if args.dry_run:
            print(dumps(manifest), end="")
            return 0
        out = args.out.resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError("--out must be new or empty; existing manifests/results are never overwritten")
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(dumps(manifest))
        if args.manifest_only:
            print(f"{len(manifest['jobs'])} job(s): {out / 'manifest.json'}")
            return 0
        return execute(manifest, cases)
    except (ValueError, OSError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
