"""JSON-driven ICMS preprocessing and bounded GPU campaigns.

    campaign = load_campaign("examples/wing_sections")
    prepared = prepare_campaign(campaign)  # HUG, reports, figures, checked meshes
    result = run_campaign(prepared)        # GPU required; no silent CPU fallback
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np

from .case import Case, CaseError, apply_override, parse_case
from .geometry.icms import Section, read_icms, write_hug
from .io.output import environment_record, git_revision
from .mesh.atlas import Atlas
from .mesh.quality import quality


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _object(value, name, allowed, required=()):
    if not isinstance(value, dict):
        raise CaseError(f"'{name}' must be an object")
    if set(value) - set(allowed):
        raise CaseError(f"'{name}': unknown keys {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        raise CaseError(f"'{name}': missing keys {sorted(set(required) - set(value))}")
    return value


def _integer(value, name, minimum):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CaseError(f"'{name}' must be an integer >= {minimum}")
    return value


def _range(value, name):
    """An explicit list or a start/stop/step range with both ends included."""
    if isinstance(value, dict):
        _object(value, name, ("start", "stop", "step"), ("start", "stop", "step"))
        numbers = [value[k] for k in ("start", "stop", "step")]
    elif isinstance(value, list) and value:
        numbers = value
    else:
        raise CaseError(f"'{name}' must be a nonempty list or a start/stop/step object")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in numbers):
        raise CaseError(f"'{name}' needs finite numbers")
    if isinstance(value, dict):
        start, stop, step = numbers
        if step <= 0 or stop < start:
            raise CaseError(f"'{name}': need start <= stop and step > 0")
        intervals = (stop - start) / step
        if not np.isfinite(intervals) or intervals > 100_000 or not np.isclose(intervals, round(intervals), atol=1e-9, rtol=1e-12):
            raise CaseError(f"'{name}': step must divide stop-start (at most 100000 intervals)")
        values = np.linspace(start, stop, round(intervals) + 1)
    else:
        values = np.asarray(numbers, dtype=float)
    if len(np.unique(values)) != len(values):
        raise CaseError(f"'{name}' contains duplicate values")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class Campaign:
    directory: Path
    document: dict
    numerical: dict
    sections: dict
    machs: tuple[float, ...]
    alphas: tuple[float, ...]
    batch_size: int
    template: Case


@dataclass(frozen=True)
class PreparedCampaign:
    campaign: Campaign
    directory: Path
    sections: tuple[Section, ...]
    cases: tuple[Case, ...]              # one prototype and mesh per section
    grids: tuple[Atlas, ...]


@dataclass(frozen=True)
class CampaignResult:
    directory: Path
    rows: tuple[dict, ...]              # fields are written per batch, not held on GPU


def load_campaign(path: str | Path, overrides: tuple[str, ...] = ()) -> Campaign:
    """Keep the simple-case schema for physics/numerics; add two strict blocks.

    Paths are relative to case.json. Mach, alpha and sweep belong to the
    campaign/geometry and cannot also be supplied in freestream.
    """
    path = Path(path).resolve()
    path = path / "case.json" if path.is_dir() else path
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise CaseError("case.json must be an object")
    numerical_path = path.parent / "numerical.json"
    numerical = json.loads(numerical_path.read_text()) if numerical_path.exists() else {}
    docs = {"case": doc, "numerical": numerical}
    for assignment in overrides:
        apply_override(docs, assignment)
    sections = _object(doc.get("sections"), "sections",
                       ("file", "count", "units", "header_lines", "plane_tolerance"),
                       ("file", "count", "units"))
    sections = {"header_lines": None, "plane_tolerance": 1e-7, **sections}
    _integer(sections["count"], "sections.count", 2)
    if not isinstance(sections["file"], str) or not sections["file"]:
        raise CaseError("sections.file must be a nonempty path")
    if sections["units"] not in ("m", "mm"):
        raise CaseError("sections.units must be 'm' or 'mm'")
    if sections["header_lines"] is not None:
        _integer(sections["header_lines"], "sections.header_lines", 0)
    tol = sections["plane_tolerance"]
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not np.isfinite(tol) or tol <= 0:
        raise CaseError("sections.plane_tolerance must be finite and positive (relative to chord)")
    schedule = _object(doc.get("campaign"), "campaign", ("mach", "alpha_deg", "batch_size"), ("mach", "alpha_deg"))
    machs, alphas = _range(schedule["mach"], "campaign.mach"), _range(schedule["alpha_deg"], "campaign.alpha_deg")
    batch_size = _integer(schedule.get("batch_size", 8), "campaign.batch_size", 1)
    simple = copy.deepcopy({k: v for k, v in doc.items() if k not in ("sections", "campaign")})
    if not isinstance(simple.get("domain"), dict) or simple["domain"].get("grid") not in ("c_grid", "o_grid"):
        raise CaseError("ICMS campaigns need domain.grid = c_grid or o_grid")
    if "airfoil" in simple["domain"]:
        raise CaseError("domain.airfoil is generated from sections.file")
    if simple["domain"].get("section", "streamwise") != "streamwise":
        raise CaseError("ICMS cuts are streamwise after flattening; domain.section must be streamwise")
    simple["domain"]["airfoil"] = "naca0012"  # schema validation only, never meshed
    stream = simple.get("freestream")
    if not isinstance(stream, dict):
        raise CaseError("freestream must be an object")
    if set(stream) & {"mach", "alpha_deg", "sweep_deg"}:
        raise CaseError("freestream mach/alpha_deg/sweep_deg are supplied by campaign and ICMS geometry")
    if stream.get("reynolds") is None:
        raise CaseError("freestream.reynolds must impose one Reynolds number for the campaign")
    simple["freestream"].update(mach=machs[0], alpha_deg=alphas[0], sweep_deg=0.0)
    template = parse_case(simple, numerical, path.parent)
    if template.sequencing:
        raise CaseError("campaign batches require numerical.convergence.grid_sequencing = 0")
    return Campaign(path.parent, copy.deepcopy(doc), numerical, sections, machs, alphas, batch_size, template)


def _case(campaign, hug, section, mach, alpha):
    doc = copy.deepcopy(campaign.template.raw["case"])
    doc["general"]["case_name"] = f"{campaign.template.name}_section_{section.index:03d}"
    doc["general"]["save_path"] = str(campaign.template.output.resolve())
    doc["domain"]["airfoil"] = str(hug)
    doc["domain"]["section"] = "streamwise"
    doc["freestream"].update(mach=mach, alpha_deg=alpha, sweep_deg=section.sweep_deg)
    return parse_case(doc, campaign.numerical, campaign.directory)


def prepare_campaign(campaign: Campaign, directory: str | Path | None = None) -> PreparedCampaign:
    """Write diagnostics and build/validate every mesh before a solver starts."""
    from .campaign_plots import plot_sections

    source = (campaign.directory / campaign.sections["file"]).resolve()
    sections = tuple(read_icms(source, campaign.sections["count"], campaign.sections["units"],
                               campaign.sections["header_lines"], campaign.sections["plane_tolerance"]))
    # Both supported cutting planes contain aircraft X. A cut normal to the
    # sweep line is a different geometry convention and must not be guessed.
    if any(abs(s.basis[0, 2]) > 1e-6 for s in sections):
        raise CaseError("ICMS cut planes must contain aircraft X (line of flight or normal to dihedral)")
    output = Path(directory).resolve() if directory is not None else (
        campaign.template.output / campaign.template.name).resolve()
    # Check the entire aerodynamic envelope before creating campaign output.
    for section in sections:
        for mach in campaign.machs:
            for alpha in campaign.alphas:
                replace(campaign.template.flow, mach=mach, alpha_deg=alpha, sweep_deg=section.sweep_deg)
    output.mkdir(parents=True, exist_ok=True)
    cases, grids, records = [], [], []
    for section in sections:
        folder = output / "sections" / f"section_{section.index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        hug = folder / "profile.hug"
        write_hug(section, hug)
        case = _case(campaign, hug, section, campaign.machs[0], campaign.alphas[0])
        grid = case.grid()
        _json(folder / "case.json", case.raw["case"])
        _json(folder / "numerical.json", campaign.numerical)
        record = {**section.record(), "hug": str(hug.relative_to(output)),
                  "grid_quality": quality(grid.mesh).record(),
                  "meshing_notes": case.grid_spec.airfoil_shape(case.directory)[1],
                  "grid_x_scale": float(np.cos(np.radians(section.sweep_deg))),
                  "solver_section": "streamwise", "alpha_correction_deg": 0.0,
                  "normalization": "LE=0; TE midpoint=(1,0); original TE gap retained"}
        _json(folder / "geometry.json", record)
        cases.append(case)
        grids.append(grid)
        records.append(record)
    plot_sections(sections, output)
    _json(output / "campaign.json", campaign.document)
    _json(output / "numerical.json", campaign.numerical)
    _json(output / "preprocessing.json", {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_units": campaign.sections["units"], "geometry_units": "m",
        "sweep_rule": "incoming quarter-chord secant; root uses outgoing secant",
        "sweep_formula": "atan2(dx, hypot(dy, dz)), degrees",
        "alpha_rule": "aircraft incidence used directly; no twist or dihedral correction",
        "reynolds_rule": "same imposed Reynolds for all sections, Mach and alpha",
        "mach": list(campaign.machs), "alpha_deg": list(campaign.alphas),
        "case_count": len(sections) * len(campaign.machs) * len(campaign.alphas),
        "batch_size": campaign.batch_size, "sections": records,
    })
    return PreparedCampaign(campaign, output, sections, tuple(cases), tuple(grids))


def run_campaign(prepared: PreparedCampaign, *, backend: str = "gpu", progress=None) -> CampaignResult:
    """Run all sections x Mach x alpha. CPU is an explicit verification option.

    Index-based paths avoid collisions between nearby floating-point angles.
    Each result is saved immediately, bounding GPU fields by batch_size.
    """
    from .campaign_batch import run_case_batch

    if backend not in ("gpu", "cpu"):
        raise ValueError("backend must be 'gpu' or 'cpu'")
    devices = jax.devices(backend)
    if not devices:
        raise RuntimeError(f"no {backend} device available; GPU campaigns never silently fall back to CPU")
    campaign = prepared.campaign
    jobs = [(s, m, a) for m in range(len(campaign.machs)) for a in range(len(campaign.alphas))
            for s in range(len(prepared.sections))]
    rows = []
    with jax.default_device(devices[0]):
        for start in range(0, len(jobs), campaign.batch_size):
            batch = jobs[start:start + campaign.batch_size]
            cases = [_case(campaign, prepared.directory / "sections" / f"section_{s:03d}" / "profile.hug",
                           prepared.sections[s], campaign.machs[m], campaign.alphas[a]) for s, m, a in batch]
            results = run_case_batch(cases, [prepared.grids[s] for s, _, _ in batch], progress)
            for (s, m, a), result in zip(batch, results):
                relative = Path("calculations") / f"section_{s:03d}" / f"mach_{m:03d}" / f"alpha_{a:03d}"
                folder = result.write(prepared.directory / relative,
                                      extra={"section_index": s, "sweep_deg": prepared.sections[s].sweep_deg,
                                             "alpha_correction_deg": 0.0})
                _json(folder / "case.json", result.case.raw["case"])
                _json(folder / "numerical.json", campaign.numerical)
                rows.append({"section": s, "mach": campaign.machs[m], "alpha_deg": campaign.alphas[a],
                             "sweep_deg": prepared.sections[s].sweep_deg, "reynolds": result.case.flow.reynolds,
                             "cl": result.forces.cl, "cd": result.forces.cd, "cm": result.forces.cm,
                             "iterations": result.history.n_iters, "exit_reason": result.history.reason,
                             "directory": str(relative)})
            # Keep an inspectable index even if a later batch is interrupted.
            _write_results(prepared.directory, rows, devices[0])
    return CampaignResult(prepared.directory, tuple(rows))


def _write_results(directory, rows, device):
    import csv

    with (directory / "campaign.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for index in sorted({row["section"] for row in rows}):
        with (directory / "sections" / f"section_{index:03d}" / "polar.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if row["section"] == index)
    _json(directory / "execution.json", {"completed_cases": len(rows), "device": str(device),
                                        "git": git_revision(), "environment": environment_record()})
