#!/usr/bin/env python3
"""Time breakdown of a continuation polar, without modifying the solver.

Subcommands:

  polar       runs ``python -m jaxrans.campaign`` unchanged (the production
              path of ``scripts/icms_continuation.py``) with timers wrapped
              around its internals (monkeypatching, read-only).
  sequence    a continuation polar through the single-case path
              (``jaxrans.simulation.run`` with ``initial`` = previous point),
              the path that works for 2.75D (sectional) cases today: the
              campaign runner does not accept sectional atlases
              (``campaign._geometry`` reads 2D wall segments).
  components  times the pieces of one Newton step on a developed transonic
              state (residual, linearization, Jacobian-vector product, line
              preconditioner factor/apply, GMRES vector work), the whole step,
              and the throughput against the batch size.

Every subcommand checks that JAX runs on a GPU (``--allow-cpu`` for smoke
tests only) and that the state lives on it, samples ``nvidia-smi``
(utilization, memory) during the run, and reports the GPU utilization inside
and outside the step blocks, and JAX's peak device memory.

Outputs: ``<out>/timing_breakdown.json`` (polar, sequence) or
``<out>/components.json``, and a printed summary.

Timers synchronize the device (``block_until_ready``) only where the solver
already waits for the result (``np.asarray`` right after each block), so the
synchronization pattern is unchanged. Step blocks are compiled ahead of time
(``lower().compile()``) by the wrapper so that compilation and execution are
reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402


def _sync(x):
    return jax.block_until_ready(x)


# ------------------------------------------------------------------ GPU checks


def require_gpu(allow_cpu: bool) -> dict:
    device = jax.devices()[0]
    info = {"default_backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "device_kind": getattr(device, "device_kind", None), "platform": device.platform}
    print(f"JAX {jax.__version__}, backend {info['default_backend']}, devices {info['devices']}")
    if device.platform != "gpu":
        if not allow_cpu:
            raise SystemExit(f"error: JAX is not running on a GPU (got {device}). Check the CUDA jaxlib "
                             "installation, or pass --allow-cpu for a functional smoke test only.")
        print("WARNING: running on CPU (--allow-cpu): timings are not representative")
    return info


def placement(tree) -> list[str]:
    """Platforms holding the leaves of ``tree`` (must be ['gpu'])."""
    platforms = set()
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "devices"):
            platforms |= {d.platform for d in leaf.devices()}
    return sorted(platforms)


def memory_stats() -> dict:
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - not every backend reports it
        stats = {}
    keys = ("peak_bytes_in_use", "bytes_in_use", "bytes_limit", "largest_alloc_size")
    return {k: int(stats[k]) for k in keys if k in stats}


class GpuSampler:
    """``nvidia-smi`` sampled in a thread: (time, utilization %, memory MiB)."""

    def __init__(self, t0: float, period_ms: int = 200):
        self.t0, self.samples, self.process, self.thread = t0, [], None, None
        if shutil.which("nvidia-smi") is None:
            return
        index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
        command = ["nvidia-smi", "-i", index, "--query-gpu=utilization.gpu,memory.used",
                   "--format=csv,noheader,nounits", f"-lms={period_ms}"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                            text=True, bufsize=1)
        except OSError:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                util, mem = (float(x) for x in line.strip().split(",")[:2])
            except ValueError:
                continue
            self.samples.append((time.perf_counter() - self.t0, util, mem))

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=5)

    def report(self, intervals) -> dict:
        if not self.samples:
            return {"available": False}
        t, util, mem = (np.asarray(x) for x in zip(*self.samples))
        inside = np.zeros(t.shape, bool)
        for start, end in intervals:
            inside |= (t >= start) & (t <= end)
        mean = lambda a: float(a.mean()) if a.size else float("nan")  # noqa: E731
        return {"available": True, "samples": int(t.size),
                "utilization_mean_%": mean(util), "utilization_in_step_blocks_%": mean(util[inside]),
                "utilization_outside_step_blocks_%": mean(util[~inside]),
                "fraction_of_samples_in_step_blocks": float(inside.mean()),
                "memory_used_peak_MiB": float(mem.max()), "memory_used_mean_MiB": float(mem.mean()),
                "note": "nvidia-smi utilization = fraction of time at least one kernel runs, "
                        "not SM occupancy: see the profiler trace (components --trace) for that"}

    def dump(self, path: Path):
        if self.samples:
            np.savetxt(path, np.asarray(self.samples), fmt="%.3f", delimiter=",",
                       header="t_s,utilization_pct,memory_MiB", comments="")


def _environment(gpu_info: dict) -> dict:
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "src"],
                                             text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"jax": jax.__version__, **gpu_info, "python": platform.python_version(),
            "git_revision": revision, "src_modified": dirty, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "jax_compilation_cache_dir": jax.config.jax_compilation_cache_dir
            if hasattr(jax.config, "jax_compilation_cache_dir") else None}


# ------------------------------------------------------------------ recorder


class Recorder:
    def __init__(self):
        self.t_start = time.perf_counter()
        self.events = defaultdict(list)      # name -> list of seconds
        self.solves = []                     # one dict per run_to_steady call
        self.points = []                     # one per finished polar point, in order
        self.groups = []                     # campaign groups (on_group)
        self.block_intervals = []            # (start, end) of step blocks, from t_start
        self.block_iterations = []           # iterations of each step block
        self.state_platforms = set()
        self.current = None

    def now(self):
        return time.perf_counter() - self.t_start

    def add(self, name, seconds):
        self.events[name].append(seconds)
        if self.current is not None:
            self.current.setdefault(name, []).append(seconds)

    def timed(self, name, fn, sync=True):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            out = fn(*args, **kwargs)
            if sync:
                _sync(out)
            self.add(name, time.perf_counter() - t)
            return out
        return wrapper


def _signature(tree):
    leaves, structure = jax.tree_util.tree_flatten(tree)
    return structure, tuple((np.shape(x), str(getattr(x, "dtype", type(x)))) for x in leaves)


class TimedBlock:
    """One jitted step block: AOT compile on first call per signature, then timed runs."""

    def __init__(self, block, recorder):
        self.block, self.recorder, self.compiled = block, recorder, {}

    def __call__(self, state):
        key = _signature(state)
        if key not in self.compiled:
            t = time.perf_counter()
            self.compiled[key] = self.block.lower(state).compile()
            self.recorder.add("compile_step_block", time.perf_counter() - t)
        start = self.recorder.now()
        out = _sync(self.compiled[key](state))
        end = self.recorder.now()
        self.recorder.add("gpu_step_blocks", end - start)
        residual = out[1][0] if isinstance(out[1], tuple) else out[1]
        self.recorder.block_iterations.append(int(np.shape(residual)[0]))
        self.recorder.block_intervals.append((start, end))
        self.recorder.state_platforms |= set(placement(out))
        return out


def install_block_probe(rec: Recorder):
    from jaxrans.solvers import driver

    # ``_masked_block`` (lru_cached, campaigns) builds its executable through
    # the module-level ``make_block``: wrapping it covers both paths and keeps
    # one TimedBlock (and its compiled cache) per executable.
    driver._masked_block.cache_clear()
    original_make_block = driver.make_block

    def make_block(*args, **kwargs):
        return TimedBlock(original_make_block(*args, **kwargs), rec)

    driver.make_block = make_block


def wrap_run_to_steady(module, rec: Recorder):
    original = module.run_to_steady

    def run_to_steady(step, state, **kwargs):
        solve = {"index": len(rec.solves), "start": rec.now()}
        rec.current = solve
        for name in ("on_block", "finite_check", "control_changed"):
            if kwargs.get(name) is not None:
                kwargs[name] = rec.timed(name, kwargs[name], sync=False)
        t = time.perf_counter()
        out = original(step, state, **kwargs)
        solve["wall"] = time.perf_counter() - t
        history = out[1]
        solve["iterations"] = int(history.n_iters)
        solve["blocks"] = len(solve.get("gpu_step_blocks", []))
        residual = np.asarray(history.residual)
        solve["cases"] = int(residual.reshape(history.n_iters, -1).shape[1]) if history.n_iters else 0
        rec.solves.append(solve)
        rec.current = None
        return out

    module.run_to_steady = run_to_steady


# ------------------------------------------------------------------ polar (campaign)


def install_campaign_probes(rec: Recorder):
    from jaxrans import campaign

    install_block_probe(rec)
    wrap_run_to_steady(campaign, rec)
    original_kernels = campaign._campaign_kernels

    def kernels(case, template):
        t = time.perf_counter()
        initialize, step, equation_monitor, monitor = original_kernels(case, template)
        rec.add("build_kernels", time.perf_counter() - t)
        calls = {"monitor": 0, "initialize": 0, "equation_monitor": 0}

        def first_split(name, fn):
            def wrapper(*args):
                t = time.perf_counter()
                out = _sync(fn(*args))
                label = f"{name}_first_call(compile)" if calls[name] == 0 else name
                calls[name] += 1
                rec.add(label, time.perf_counter() - t)
                return out
            return wrapper

        return (first_split("initialize", initialize), step,
                first_split("equation_monitor", equation_monitor), first_split("monitor", monitor))

    campaign._campaign_kernels = kernels
    campaign.integrate_forces = rec.timed("integrate_forces", campaign.integrate_forces)
    original_run_campaign = campaign.run_campaign

    def run_campaign(*args, **kwargs):
        on_point, on_group = kwargs.get("on_point"), kwargs.get("on_group")

        def point(name, alpha, result):
            t = time.perf_counter()
            if on_point is not None:
                on_point(name, alpha, result)
            rec.add("save_point", time.perf_counter() - t)
            rec.points.append(_point_record(name, alpha, result))

        def group(requests):
            rec.groups.append({"time": rec.now(), "profiles": [r.name for r in requests]})
            if on_group is not None:
                on_group(requests)

        kwargs["on_point"], kwargs["on_group"] = point, group
        return original_run_campaign(*args, **kwargs)

    campaign.run_campaign = run_campaign
    original_write = campaign.Campaign.write

    def write(self, *args, **kwargs):
        t = time.perf_counter()
        out = original_write(self, *args, **kwargs)
        rec.add("final_write", time.perf_counter() - t)
        return out

    campaign.Campaign.write = write
    return campaign


def _point_record(name, alpha, result) -> dict:
    return {"profile": name, "alpha": float(alpha), "iterations": int(result.history.n_iters),
            "point_wall_s": float(result.history.seconds), "chain": result.provenance.get("chain"),
            "exit_reason": result.history.reason, "cl": float(result.forces.cl),
            "cd": float(result.forces.cd), "cm": float(result.forces.cm)}


def run_polar(args) -> int:
    gpu = require_gpu(args.allow_cpu)
    rec = Recorder()
    sampler = GpuSampler(rec.t_start)
    campaign = install_campaign_probes(rec)
    manifest = args.manifest
    if args.alpha is not None:
        doc = json.loads(manifest.read_text())
        for entry in doc["profiles"]:
            entry.pop("alphas", None)
            entry["alpha"] = list(args.alpha)
        manifest = manifest.resolve().parent / f".{manifest.stem}_bench_override.json"
        manifest.write_text(json.dumps(doc, indent=2))
    argv = [str(manifest), "--out", str(args.out)] + (["--quiet"] if args.quiet else [])
    if getattr(args, "chains", None) is not None:
        argv += ["--chains", str(args.chains)]
    if getattr(args, "scheduler", None) is not None:
        argv += ["--scheduler", args.scheduler]
    status = 1
    try:
        status = campaign.main(argv)
    finally:
        if manifest.name.endswith("_bench_override.json"):
            manifest.unlink(missing_ok=True)
        sampler.stop()
        finish(rec, sampler, gpu, args, {"mode": "campaign", "manifest": str(args.manifest),
                                         "alpha_override": args.alpha})
    return status


# ------------------------------------------------------------------ sequence (single-case path)


def run_sequence(args) -> int:
    gpu = require_gpu(args.allow_cpu)
    rec = Recorder()
    sampler = GpuSampler(rec.t_start)
    from jaxrans import load_case, simulation
    from jaxrans.polar import alpha_range, at_alpha
    from jaxrans.solvers.driver import ExitReason

    install_block_probe(rec)
    wrap_run_to_steady(simulation, rec)
    simulation.integrate_forces = rec.timed("integrate_forces", simulation.integrate_forces)
    simulation.make_stepper = rec.timed("make_stepper(trace setup)", simulation.make_stepper, sync=False)
    status = 1
    try:
        case = load_case(args.case, tuple(args.overrides))
        grid = case.grid()
        rec.groups.append({"time": rec.now(), "profiles": [case.name]})
        alphas = alpha_range(*args.alpha)
        print(f"{case.name}: {alphas.size} angles {alphas[0]:+g}..{alphas[-1]:+g}, canvas {grid.shape}")
        previous = None
        for alpha in alphas:
            sub = at_alpha(case, alpha)
            result = simulation.run(sub, grid, initial=previous)
            t = time.perf_counter()
            if not args.no_save:
                result.write(args.out / f"alpha_{alpha:+.2f}")
            rec.add("save_point", time.perf_counter() - t)
            rec.points.append(_point_record(case.name, alpha, result))
            p = rec.points[-1]
            print(f"alpha {alpha:+6.2f}: {p['iterations']:5d} it, {p['exit_reason']:<9s} "
                  f"CL {p['cl']:+.5f} CD {p['cd']:.6f} ({rec.solves[-1]['wall']:.1f} s)", flush=True)
            if result.history.reason not in (ExitReason.CONVERGED, ExitReason.SETTLED):
                print("stopping: the continuation point failed (as the campaign runner does)")
                break
            previous = result.U
        status = 0
    finally:
        sampler.stop()
        finish(rec, sampler, gpu, args, {"mode": "sequence", "case": str(args.case),
                                         "overrides": args.overrides, "alpha": args.alpha})
    return status


# ------------------------------------------------------------------ report


def finish(rec, sampler, gpu, args, meta):
    total = rec.now()
    report = summarize(rec, total, sampler, gpu, meta)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timing_breakdown.json").write_text(json.dumps(report, indent=2, default=float))
    sampler.dump(args.out / "gpu_samples.csv")
    print_report(report)
    print(f"\nwritten to {args.out / 'timing_breakdown.json'}")


def summarize(rec: Recorder, total: float, sampler, gpu, meta) -> dict:
    """Scheduler-agnostic: step blocks are timed where they run, everything
    else by its own probe; what no probe covers is reported as host time."""
    s = {k: float(np.sum(v)) for k, v in rec.events.items()}
    n = {k: len(v) for k, v in rec.events.items()}
    preparation = rec.groups[0]["time"] if rec.groups else float("nan")
    gpu_time = s.get("gpu_step_blocks", 0.0)
    compile_step = s.get("compile_step_block", 0.0)
    monitor = sum(v for k, v in s.items() if k.startswith(("monitor", "equation_monitor", "initialize")))
    save, forces, final = s.get("save_point", 0.0), s.get("integrate_forces", 0.0), s.get("final_write", 0.0)
    on_block = s.get("on_block", 0.0)
    if rec.solves:                     # rounds scheduler / sequence: monitoring runs inside on_block
        monitor_or_on_block = on_block + sum(v for k, v in s.items() if k.startswith(("initialize", "equation_monitor")))
        forces_outside = forces if meta.get("mode") != "sequence" else 0.0
    else:
        monitor_or_on_block, forces_outside = monitor, forces
    host = total - preparation - compile_step - gpu_time - monitor_or_on_block - save - forces_outside - final
    breakdown = {
        "1_preparation(load, grid, wall distance)": preparation,
        "2_compile_step_block": compile_step,
        "3_gpu_step_blocks": gpu_time,
        "4_monitoring(equation norms, forces per block, initial states)": monitor_or_on_block,
        "5_save_point": save,
        "6_integrate_forces(end of points)": forces_outside,
        "7_final_write": final,
        "8_other_host(scheduling, callbacks, numpy, tracing)": host,
        "total_wall": total,
    }
    batch_iters = int(np.sum(rec.block_iterations))
    case_iters = sum(p["iterations"] for p in rec.points)
    platforms = sorted(rec.state_platforms)
    return {
        "environment": _environment(gpu), **meta,
        "state_on_platforms": platforms, "state_on_gpu": platforms == ["gpu"],
        "breakdown_s": breakdown,
        "gpu_step_blocks_fraction_of_total": gpu_time / total if total else float("nan"),
        "step_blocks": len(rec.block_iterations), "batch_iterations": batch_iters, "case_iterations": case_iters,
        "ms_per_batch_iteration_gpu": 1e3 * gpu_time / batch_iters if batch_iters else float("nan"),
        "ms_per_case_iteration_gpu": 1e3 * gpu_time / case_iters if case_iters else float("nan"),
        "nvidia_smi": sampler.report(rec.block_intervals), "jax_memory": memory_stats(),
        "call_counts": n, "event_totals_s": s, "groups": rec.groups, "points": rec.points,
    }


def print_report(r: dict):
    print("\n================ time breakdown ================")
    total = r["breakdown_s"]["total_wall"]
    for k, v in r["breakdown_s"].items():
        print(f"{k:<66s} {v:10.2f} s  {100 * v / total if total else 0:6.1f} %")
    print(f"step blocks / total wall : {100 * r['gpu_step_blocks_fraction_of_total']:.1f} %")
    print(f"{r['step_blocks']} blocks, batch iterations {r['batch_iterations']}, case iterations "
          f"{r['case_iterations']}: {r['ms_per_batch_iteration_gpu']:.2f} ms per batch iteration, "
          f"{r['ms_per_case_iteration_gpu']:.2f} ms per case iteration (GPU blocks)")
    print(f"state on {r['state_on_platforms']}  ->  {'OK, on GPU' if r['state_on_gpu'] else 'NOT ON GPU'}")
    g = r["nvidia_smi"]
    if g.get("available"):
        print(f"nvidia-smi: utilization mean {g['utilization_mean_%']:.0f} %, in step blocks "
              f"{g['utilization_in_step_blocks_%']:.0f} %, outside {g['utilization_outside_step_blocks_%']:.0f} %; "
              f"memory peak {g['memory_used_peak_MiB']:.0f} MiB")
    else:
        print("nvidia-smi: not available")
    if r["jax_memory"]:
        print(f"JAX device memory: peak {r['jax_memory'].get('peak_bytes_in_use', 0) / 2**20:.0f} MiB")
    print(f"\n{'profile':<24s} {'chain':>5s} {'alpha':>6s} {'iters':>6s} {'exit':>9s} {'point s':>8s} "
          f"{'CL':>9s} {'CD':>9s}")
    for p in r["points"]:
        chain = "" if p.get("chain") is None else str(p["chain"])
        print(f"{p['profile'][:24]:<24s} {chain:>5s} {p['alpha']:+6.2f} {p['iterations']:6d} "
              f"{p['exit_reason'][:9]:>9s} {p['point_wall_s']:8.1f} {p['cl']:+9.5f} {p['cd']:9.6f}")


# ------------------------------------------------------------------ components


def _time(fn, *args, repeat=5):
    _sync(fn(*args))                      # compile + warm
    runs = []
    for _ in range(repeat):
        t = time.perf_counter()
        _sync(fn(*args))
        runs.append(time.perf_counter() - t)
    return 1e3 * float(np.median(runs))


def run_components(args) -> int:
    gpu = require_gpu(args.allow_cpu)
    t0 = time.perf_counter()
    sampler = GpuSampler(t0)
    from jaxrans import load_case
    from jaxrans.core.state import freestream_primitive
    from jaxrans.numerics.residual import make_residual
    from jaxrans.polar import at_alpha
    from jaxrans.simulation import make_stepper
    from jaxrans.solvers.krylov import gmres_right
    from jaxrans.solvers.preconditioners import get_preconditioner

    case = load_case(args.case, tuple(args.overrides))
    if case.numerics.solver != "implicit":
        raise SystemExit("an implicit case is required")
    grid = case.grid(coarsen=args.coarsen)
    sub = at_alpha(case, args.alpha_deg)
    q = jnp.asarray(freestream_primitive(sub.flow, grid.components), dtype=case.numerics.dtype)
    distance = make_residual(grid, case.flow, case.numerics).distance
    step, state = make_stepper(case, grid, freestream=q, distance=distance)
    out = {"environment": _environment(gpu), "case": str(args.case), "overrides": args.overrides,
           "alpha_deg": args.alpha_deg, "canvas_shape": list(grid.shape),
           "canvas_cells": int(np.prod(grid.shape)), "interior_cells": int(np.sum(np.asarray(grid.interior))),
           "nvar": int(state.U.shape[0]), "dtype": str(state.U.dtype),
           "implicit": {k: getattr(case.implicit, k) for k in (
               "krylov_subspace", "krylov_restarts", "max_relative_update", "backtracking_steps",
               "nonlinear_growth_limit", "preconditioner", "jacobian")}}
    print(f"canvas {out['canvas_shape']} ({out['canvas_cells']} cells, {out['interior_cells']} interior), "
          f"nvar {out['nvar']}, {out['dtype']}")

    # 1. Develop a transonic state.
    n = args.block_iters

    def block(s):
        return lax.scan(lambda c, _: step(c), s, None, length=n)

    block_j = jax.jit(block)
    t = time.perf_counter()
    for _ in range(max(1, args.warmup // n)):
        state, residuals = block_j(state)
    _sync(state)
    out["state_on_platforms"] = placement(state)
    out["warmup"] = {"iterations": max(1, args.warmup // n) * n, "seconds_incl_compile": time.perf_counter() - t,
                     "cfl": float(state.cfl), "residual_last": float(np.asarray(residuals)[-1])}
    print(f"warm-up: {out['warmup']['iterations']} iterations, CFL {out['warmup']['cfl']:.1f}, "
          f"state on {out['state_on_platforms']}")

    # 2. Whole step, single case.
    b0 = time.perf_counter() - t0
    ms_step = _time(block_j, state, repeat=3) / n
    step_interval = (b0, time.perf_counter() - t0)
    out["step_ms_single_case"] = ms_step

    # 3. Components on the developed state.
    U, cfl = state.U, state.cfl
    op = make_residual(grid, case.flow, case.numerics, freestream=q, distance=distance)
    jac = op if case.implicit.jacobian == "exact" else make_residual(
        grid, case.flow, case.numerics, first_order=True, freestream=q, distance=distance)
    prepare = get_preconditioner(case.implicit.preconditioner)(op.atlas, op.gamma, op.viscosity, op.model, op.distance)
    k = case.implicit.krylov_subspace + 1
    key = jax.random.PRNGKey(0)
    V = jax.random.normal(key, (k,) + U.shape, dtype=U.dtype) * jnp.abs(U).mean()

    def jvps(U, V):
        _, f = jax.linearize(jac.residual, U)
        return lax.scan(lambda c, v: (c, f(v)), 0, V)[1]

    def applies(U, cfl, R):
        P = prepare(U, op.wave_speeds(U), cfl)
        return lax.scan(lambda c, r: (c, P(r)), 0, R)[1]

    jvps_j, applies_j = jax.jit(jvps), jax.jit(applies)
    c = {"residual_2nd_order": _time(jax.jit(op.residual), U), "wave_speeds": _time(jax.jit(op.wave_speeds), U)}
    t1, tk = _time(jvps_j, U, V[:1]), _time(jvps_j, U, V)
    c["jvp_1st_order(each)"] = (tk - t1) / (k - 1)
    c["linearize_1st_order"] = t1 - c["jvp_1st_order(each)"]
    t1, tk = _time(applies_j, U, cfl, V[:1]), _time(applies_j, U, cfl, V)
    c["precond_apply(each)"] = (tk - t1) / (k - 1)
    c["precond_factor"] = t1 - c["precond_apply(each)"] - c["wave_speeds"]
    restart, cycles = case.implicit.krylov_subspace, case.implicit.krylov_restarts
    d = 1.0 + jnp.abs(jax.random.normal(key, U.shape, dtype=U.dtype))
    gm = jax.jit(lambda b: gmres_right(lambda v: d * v, b, lambda r: r, restart, cycles, 1e-30))
    c["gmres_vector_work(all cycles)"] = _time(gm, V[0])
    out["components_ms"] = c

    extra_mv = 1 if case.implicit.max_relative_update is not None else 0
    model = {}
    for used in range(1, cycles + 1):
        per_cycle = (restart + 1) * (c["jvp_1st_order(each)"] + c["precond_apply(each)"])
        predicted = (c["residual_2nd_order"] + c["linearize_1st_order"] + c["wave_speeds"] + c["precond_factor"]
                     + used * per_cycle + extra_mv * c["jvp_1st_order(each)"]
                     + c["gmres_vector_work(all cycles)"] * used / cycles)
        model[f"{used}_cycle(s)"] = {
            "jvp_count": used * (restart + 1) + extra_mv, "precond_apply_count": used * (restart + 1),
            "jvp_total": (used * (restart + 1) + extra_mv) * c["jvp_1st_order(each)"],
            "precond_apply_total": used * (restart + 1) * c["precond_apply(each)"],
            "predicted_step": predicted, "unexplained(measured - predicted)": ms_step - predicted}
    out["step_model_ms"] = model

    # 4. Batch scaling (vmap over identical cases, as a polar batch does).
    scaling = []
    for b in args.batch:
        try:
            qs = jnp.broadcast_to(q, (b,) + q.shape)
            states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (b,) + x.shape), state)

            def vblock(s, qs):
                def one(c, _):
                    return jax.vmap(lambda si, fi: make_stepper(case, grid, freestream=fi,
                                                                distance=distance)[0](si))(c, qs)
                return lax.scan(one, s, None, length=n)

            vb = jax.jit(vblock)
            ms = _time(vb, states, qs, repeat=3) / n
            scaling.append({"batch": b, "ms_per_batch_iteration": ms, "ms_per_case_iteration": ms / b,
                            "jax_memory": memory_stats()})
            print(f"batch {b:3d}: {ms:8.2f} ms/iteration, {ms / b:7.2f} ms/case-iteration")
        except Exception as error:  # noqa: BLE001 - typically out of memory
            scaling.append({"batch": b, "error": str(error).splitlines()[0][:300]})
            print(f"batch {b}: failed ({str(error).splitlines()[0][:120]})")
            break
    out["batch_scaling"] = scaling

    if args.trace:
        args.trace.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(args.trace)):
            for _ in range(2):
                state, _r = block_j(state)
            _sync(state)
        out["trace"] = str(args.trace)

    sampler.stop()
    out["nvidia_smi_single_case_step"] = sampler.report([step_interval])
    out["jax_memory"] = memory_stats()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "components.json").write_text(json.dumps(out, indent=2, default=float))
    sampler.dump(args.out / "gpu_samples.csv")
    print(f"\nwhole Newton step (1 case): {ms_step:.2f} ms")
    for name, v in c.items():
        print(f"  {name:<34s} {v:9.3f} ms")
    for name, m in model.items():
        print(f"  model {name}: predicted {m['predicted_step']:.2f} ms "
              f"(JVP {m['jvp_total']:.2f}, precond apply {m['precond_apply_total']:.2f}), "
              f"unexplained {m['unexplained(measured - predicted)']:+.2f} ms")
    g = out["nvidia_smi_single_case_step"]
    if g.get("available"):
        print(f"nvidia-smi during the single-case step timing: {g['utilization_in_step_blocks_%']:.0f} % utilization")
    print(f"\nwritten to {args.out / 'components.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("polar", help="instrumented continuation campaign (jaxrans.campaign)")
    p.add_argument("manifest", type=Path)
    p.add_argument("--alpha", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                   help="override the manifest's alpha range (e.g. a quick 0 3 1)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--chains", type=int, help="continuation chains per profile (jaxrans.campaign --chains)")
    p.add_argument("--scheduler", choices=("slots", "rounds"),
                   help="campaign scheduler (default: the campaign's); 'rounds' is the previous behaviour")
    s = sub.add_parser("sequence", help="instrumented continuation through the single-case path")
    s.add_argument("case", type=Path)
    s.add_argument("--alpha", type=float, nargs=3, metavar=("START", "STOP", "STEP"), required=True)
    s.add_argument("--set", dest="overrides", action="append", default=[])
    s.add_argument("--no-save", action="store_true", help="do not write the per-point result directories")
    c = sub.add_parser("components", help="per-component timing of a Newton step and batch scaling")
    c.add_argument("case", type=Path)
    c.add_argument("--set", dest="overrides", action="append", default=[])
    c.add_argument("--alpha-deg", type=float, default=2.0)
    c.add_argument("--warmup", type=int, default=300, help="Newton iterations before timing")
    c.add_argument("--block-iters", type=int, default=20)
    c.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    c.add_argument("--coarsen", type=int, default=0, help="CPU smoke tests only")
    c.add_argument("--trace", type=Path, help="also write a JAX profiler trace of two blocks here")
    for command in (p, s, c):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--allow-cpu", action="store_true", help="functional smoke test on CPU only")
    args = parser.parse_args()
    return {"polar": run_polar, "sequence": run_sequence, "components": run_components}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
