"""A polar: one case at many angles of attack, batched on the GPU(s) by default.

    python -m jaxrans.polar CASE_DIR --alpha START STOP STEP [--devices N]
                            [--mode batch|continuation]
                            [--set file.section.key=value ...] [--out DIR] [--quiet]

The angle of attack enters the solver only through the freestream vector
``(rho, u, v, p)``, which ``make_residual`` accepts as an array. So one step is
traced with the freestream as an argument and ``jax.vmap``-ed over the angles:
every kernel then does all angles' work at once, which fills a GPU that one
case leaves latency-bound (the line preconditioner's sequential sweep). Each
angle keeps its own CFL ramp, rejections and convergence; they never interact.
The driver (``solvers/driver.py``) marches the batch and splits it over the
devices.

With ``--mode continuation`` the angles run sequentially in requested order,
each initialized from the previous converged/settled solution. For several
profiles batched together with continuation, use ``python -m jaxrans.campaign``.

Output: ``<out>/polar.dat`` (alpha, CL, CD, friction, CM, orders, iterations,
CL spread, settled), ``<out>/polar_metadata.json`` (case, timing, provenance),
and one ordinary result directory per angle, ``<out>/alpha_+2.50/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .mesh.atlas import Atlas
from .core.state import freestream_primitive
from .numerics.residual import make_residual
from .case import Case, CaseError, load_case
from .post.forces import integrate_forces, pressure_lift
from .io.output import environment_record, git_revision
from .simulation import SETTLED_CL_SPREAD, Result, Timing, make_stepper
from .solvers.driver import run_to_steady


@dataclass(frozen=True)
class ContinuationTiming:
    """Sequential rounds; iterations count physical profile steps, not batch steps."""

    total: float
    iterations: int
    cases: int = 1
    devices: int = 1

    def record(self) -> dict:
        return {"total_s": self.total, "case_iterations": self.iterations,
                "cases": self.cases, "devices": self.devices, "mode": "continuation"}

    def summary(self) -> str:
        return f"{self.total:.1f} s, {self.iterations} case iterations (continuation)"


@dataclass(frozen=True)
class Polar:
    """One ``Result`` per angle, and the timing of the batch."""

    case: Case
    alphas: np.ndarray
    results: list[Result]
    timing: Timing | ContinuationTiming
    mode: str = "batch"
    requested_alphas: tuple[float, ...] = ()
    attempts: tuple[Result, ...] = ()
    continuation: dict | None = None

    def table(self) -> np.ndarray:
        """Rows ``alpha, CL, CD, CD friction, CM, orders, iterations, CL spread, settled``."""
        return np.array([[a, r.forces.cl, r.forces.cd, r.forces.cd_friction, r.forces.cm,
                          r.history.orders_dropped, r.history.n_iters, r.cl_spread,
                          float(r.history.reason in ("converged", "settled") if self.mode == "continuation"
                                else r.cl_spread <= SETTLED_CL_SPREAD)]
                         for a, r in zip(self.alphas, self.results)]).reshape(-1, 9)

    def write(self, directory: str | Path, *, preserve_prefix: int = 0) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.continuation is not None and (directory / "continuation.json").exists():
            previous = json.loads((directory / "continuation.json").read_text())
            if (previous.get("case_setup") != self.case.raw or
                    previous.get("continuation") != self.continuation or
                    previous.get("requested_alphas") !=
                    list(self.requested_alphas)[:len(previous.get("requested_alphas", []))]):
                raise ValueError(f"{directory}: adaptive output configuration differs; use a new directory")
        if self.continuation is not None:
            from .campaign import _atomic_json, _checkpoint_attempt
            for result in self.attempts:
                _checkpoint_attempt(directory, result)
        np.savetxt(directory / "polar.dat", self.table(),
                   header="alpha_deg cl cd cd_friction cm orders_dropped iterations cl_spread settled",
                   fmt=["%.4f", "%.8f", "%.8f", "%.8f", "%.8f", "%.3f", "%d", "%.3e", "%d"])
        (directory / "polar_metadata.json").write_text(json.dumps({
            "case": self.case.name, "case_setup": self.case.raw, "alphas": self.alphas.tolist(),
            "timing": self.timing.record(), "settled_cl_spread": SETTLED_CL_SPREAD,
            "mode": self.mode, "requested_alphas": list(self.requested_alphas) or self.alphas.tolist(),
            "points": [{"alpha": float(a), "exit_reason": r.history.reason,
                        "iterations": r.history.n_iters,
                        "initial_alpha": float(self.alphas[k - 1]) if self.mode == "continuation" and k else None}
                       for k, (a, r) in enumerate(zip(self.alphas, self.results))],
            "git": git_revision(), "environment": environment_record(),
        }, indent=2, default=float))
        if self.continuation is not None:
            _atomic_json(directory / "continuation.json", {
                "version": 1, "case_setup": self.case.raw,
                "continuation": self.continuation, "requested_alphas": list(self.requested_alphas),
            })
            meta_path = directory / "polar_metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["continuation"] = self.continuation
            meta["attempts"] = [{**r.provenance, "exit_reason": r.history.reason,
                                 "directory": f"attempts/attempt_{r.provenance['attempt']:06d}"}
                                for r in self.attempts]
            for point, result in zip(meta["points"], self.results):
                point.update(result.provenance)
                point["directory"] = f"attempts/attempt_{result.provenance['attempt']:06d}"
            _atomic_json(meta_path, meta)
            return directory
        names = [f"alpha_{a:+.2f}" for a in self.alphas]
        unique_names = len(set(names)) == len(names)
        for k, (alpha, result) in enumerate(zip(self.alphas, self.results)):
            if k < preserve_prefix:
                continue
            name = names[k] if unique_names else f"point_{k:04d}_alpha_{alpha:+.8g}"
            result.write(directory / name, extra={
                "polar_mode": self.mode,
                "initial_alpha": float(self.alphas[k - 1]) if self.mode == "continuation" and k else None,
            })
        return directory


def at_alpha(case: Case, alpha: float) -> Case:
    flow = dataclasses.replace(case.flow, alpha_deg=float(alpha))
    return dataclasses.replace(case, flow=flow, name=f"{case.name}_a{alpha:+.2f}")


def run_polar(case: Case, alphas, grid: Atlas | None = None, devices: int | None = None,
              progress=None, *, mode: str = "batch") -> Polar:
    """Compute angles in a batch (default) or in requested-order continuation.

    Continuation stops at its first failed point and retains partial results.
    ``progress(iteration, CL per angle)`` is called per block; continuation
    reports a one-entry CL array for its current angle.
    """
    alphas = validate_alphas(alphas)
    if devices is not None and (not isinstance(devices, int) or devices < 1 or devices > len(jax.devices())):
        raise ValueError(f"devices must be between 1 and {len(jax.devices())}")
    if mode == "continuation":
        from .campaign import PolarRequest, run_campaign

        def report(profile, alpha, iteration, cl):
            if progress is not None:
                progress(iteration, np.array([cl]))

        campaign = run_campaign([PolarRequest("profile", case, tuple(alphas), grid)],
                                devices=devices, progress=report)
        return campaign.polars["profile"]
    if mode != "batch":
        raise ValueError("mode must be 'batch' or 'continuation'")
    grid = case.grid() if grid is None else grid
    use = jax.devices()[: devices or len(jax.devices())]
    # Pad to a multiple of the device count with copies of the last angle.
    padded = np.concatenate([alphas, np.repeat(alphas[-1:], -alphas.size % len(use))])
    freestream = jnp.asarray([freestream_primitive(at_alpha(case, a).flow, grid.components) for a in padded],
                             dtype=case.numerics.dtype)
    distance = make_residual(grid, case.flow, case.numerics).distance

    def stepper(q):
        return make_stepper(case, grid, freestream=q, distance=distance)

    def step(carry):
        states, q = carry
        states, residuals = jax.vmap(lambda s, f: stepper(f)[0](s))(states, q)
        return (states, q), residuals

    lift = jax.jit(jax.vmap(lambda U, q: pressure_lift(U, grid, q, case.flow.gamma)))
    lifts: list[np.ndarray] = []

    def on_block(index, carry, residuals):
        lifts.append(np.asarray(lift(carry[0].U, carry[1]))[: alphas.size])
        if progress is not None:
            progress((index + 1) * case.block_iters, lifts[-1])

    carry, history = run_to_steady(
        step, (jax.jit(jax.vmap(lambda f: stepper(f)[1]))(freestream), freestream),
        max_iters=case.numerics.max_iters, tol=case.numerics.tol, block_iters=case.block_iters,
        on_block=on_block, devices=use)

    timing = Timing.of(history, case.block_iters, grid.cells, cases=alphas.size, devices=len(use))
    U, lifts = np.asarray(carry[0].U), np.array(lifts)
    results = []
    for k, alpha in enumerate(alphas):
        sub, Uk = at_alpha(case, alpha), jnp.asarray(U[k])
        forces = integrate_forces(Uk, grid, sub.flow, case.grid_spec.force_reference())
        results.append(Result(sub, grid, Uk, forces, history.case(k), lifts[:, k], timing))
    return Polar(case, alphas, results, timing)


def validate_alphas(alphas) -> np.ndarray:
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or not alphas.size or not np.all(np.isfinite(alphas)) or np.any(np.abs(alphas) > 90):
        raise ValueError("alphas must be a nonempty finite vector in [-90, 90] degrees")
    return alphas


def alpha_range(start: float, stop: float, step: float) -> np.ndarray:
    if not np.all(np.isfinite([start, stop, step])) or step == 0 or (stop - start) * step < 0:
        raise ValueError("alpha range needs a nonzero step directed from start toward stop")
    if abs(start) > 90 or abs(stop) > 90:
        raise ValueError("alpha range must lie in [-90, 90] degrees")
    count = int(np.floor((stop - start) / step + 1e-10)) + 1
    return validate_alphas(np.round(start + step * np.arange(count), 10))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jaxrans.polar", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="case directory (case.json + numerical.json)")
    parser.add_argument("--alpha", type=float, nargs=3, required=True, metavar=("START", "STOP", "STEP"),
                        help="angles START..STOP (included) by STEP, degrees")
    parser.add_argument("--devices", type=int, help="devices to split the angles over (default: all)")
    parser.add_argument("--mode", choices=("batch", "continuation"), default="batch",
                        help="batch angles independently, or initialize each angle from the previous solution")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="FILE.SECTION.KEY=VALUE", help="override one key; repeatable")
    parser.add_argument("--out", type=Path, help="output directory (default: <save_path>/<case_name>_polar)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        case = load_case(args.case, tuple(args.overrides))
    except (CaseError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    start, stop, step = args.alpha
    try:
        alphas = alpha_range(start, stop, step)
    except ValueError as error:
        parser.error(str(error))
    print(f"{case.name}: {alphas.size} angles {start:+g}..{stop:+g} deg, mode {args.mode}")

    def progress(iteration, cl):
        print(f"{iteration:8d}  CL " + " ".join(f"{c:+.4f}" for c in cl))

    try:
        polar = run_polar(case, alphas, devices=args.devices, progress=None if args.quiet else progress,
                          mode=args.mode)
    except ValueError as error:
        parser.error(str(error))
    out = polar.write(args.out or case.output / f"{case.name}_polar")
    print("   alpha        CL          CD   orders  CL spread  settled  exit reason")
    for row, result in zip(polar.table(), polar.results):
        print(f"{row[0]:+8.2f}  {row[1]:+.6f}  {row[2]:+.7f}  {row[5]:6.2f}  {row[7]:9.1e}  "
              f"{bool(row[8])}  {result.history.reason}")
    print(f"{polar.timing.summary()}\nwritten to {out}")
    if args.mode == "continuation":
        from .solvers.driver import ExitReason
        return int(any(r.history.reason not in (ExitReason.CONVERGED, ExitReason.SETTLED) for r in polar.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
