"""Frozen configs, validated once at construction so the numerics need no
input checks. Hashable and static under ``jit``; what a batch varies (a
polar's freestream) travels as an array instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants


TURBULENCE_MODELS: tuple[str, ...] = ("sa_noft2", "sa_noft2_qcr")
"""Spalart-Allmaras without the ft2 term, with SA-neg, and the same with the
quadratic constitutive relation QCR2000 (physics/spalart_allmaras.py)."""


@dataclass(frozen=True)
class FlowConditions:
    """Freestream and gas, nondimensional (constants.py). ``reynolds=None`` is
    inviscid; ``sutherland_ratio`` is ``S / T_inf`` (``None``: constant
    viscosity); ``nu_tilde_ratio`` is ``nu~_inf / nu_inf``.

    ``sweep_deg`` makes a 2D grid an infinite swept wing (2.5D): the grid is the
    section normal to the sweep line, ``alpha`` is streamwise, and the velocity
    gains a spanwise component (core/state.py, ``freestream_vector``).
    ``None``, the default, is plain 2D; ``0.0`` the 2.5D path without sweep."""

    mach: float
    alpha_deg: float = 0.0
    reynolds: float | None = None
    gamma: float = constants.GAMMA_AIR
    sutherland_ratio: float | None = None
    turbulence: str | None = None
    nu_tilde_ratio: float = 3.0
    sweep_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.mach > 0.0:
            raise ValueError(f"mach must be positive, got {self.mach}")
        if self.sweep_deg is not None and not abs(self.sweep_deg) < 90.0:
            raise ValueError(f"sweep_deg must lie in (-90, 90), got {self.sweep_deg}")
        if self.in_plane_mach >= 1.0:
            # Subsonic/transonic freestream only: the characteristic far-field
            # BC assumes a subsonic outer boundary. Swept, its characteristics
            # are in the grid's plane: the in-plane Mach number counts.
            raise ValueError(
                f"freestream mach must be subsonic for the characteristic "
                f"far-field BC, got {self.mach}"
                + (f" (in-plane {self.in_plane_mach:.4g})" if self.swept else "")
            )
        if not self.gamma > 1.0:
            raise ValueError(f"gamma must exceed 1, got {self.gamma}")
        if self.reynolds is not None and not self.reynolds > 0.0:
            raise ValueError(f"reynolds must be positive or None, got {self.reynolds}")
        if self.sutherland_ratio is not None:
            if self.reynolds is None:
                raise ValueError("sutherland_ratio given for an inviscid case")
            if not self.sutherland_ratio > 0.0:
                raise ValueError(f"sutherland_ratio must be positive, got {self.sutherland_ratio}")
        if self.turbulence is not None:
            if self.turbulence not in TURBULENCE_MODELS:
                raise ValueError(f"turbulence must be one of {TURBULENCE_MODELS}, "
                                 f"got {self.turbulence!r}")
            if self.reynolds is None:
                raise ValueError("a turbulence model needs a Reynolds number")
        if not self.nu_tilde_ratio > 0.0:
            raise ValueError(f"nu_tilde_ratio must be positive, got {self.nu_tilde_ratio}")
        if abs(self.alpha_deg) > 90.0:
            raise ValueError(f"alpha_deg out of range, got {self.alpha_deg}")

    @property
    def inviscid(self) -> bool:
        return self.reynolds is None

    @property
    def alpha_rad(self) -> float:
        return float(np.deg2rad(self.alpha_deg))

    @property
    def swept(self) -> bool:
        """An infinite swept wing on a 2D grid (2.5D)."""
        return self.sweep_deg is not None

    @property
    def sweep_rad(self) -> float | None:
        return None if self.sweep_deg is None else float(np.deg2rad(self.sweep_deg))

    @property
    def in_plane_mach(self) -> float:
        """Mach number of the velocity in the grid's plane: ``M`` unswept,
        ``M sqrt(cos^2 alpha cos^2 Lambda + sin^2 alpha)`` swept."""
        if not self.swept:
            return self.mach
        a, sweep = self.alpha_rad, self.sweep_rad
        return float(self.mach * np.hypot(np.cos(a) * np.cos(sweep), np.sin(a)))

    @property
    def mu_inf(self) -> float:
        """``mu = M / Re``, at the freestream temperature."""
        if self.reynolds is None:
            raise ValueError("mu_inf is undefined for an inviscid case")
        return self.mach / self.reynolds


SOLVERS: tuple[str, ...] = ("explicit", "implicit")
"""Pseudo-time paths: multistage Runge-Kutta, pseudo-transient Newton-Krylov."""

PRECISIONS = {"fp64": "float64", "float64": "float64", "fp32": "float32", "float32": "float32"}


@dataclass(frozen=True)
class NumericsConfig:
    """Discretization and stepping (``docs/case_reference.md``). Each normalized
    equation residual must fall below its relative peak threshold or absolute
    floor; force stability is additionally required when ``forces_tol`` is set.
    With ``forces_tol=None``, ``tol`` applies to every equation without an
    absolute floor. ``far_field`` keys ``numerics.boundary.FAR_FIELDS``."""

    flux: str = "hllc"
    limiter: str = "minmod"
    cfl: float = 1.5
    max_iters: int = 40_000
    tol: float = 1.0e-12
    forces_tol: float | None = 1.0e-4
    equation_tol: float = 1.0e-4
    equation_abs_tol: float = 1.0e-8
    n_ghost: int = 2
    solver: str = "explicit"
    far_field: str = "characteristic"
    kappa: float = 0.0
    precision: str = "fp64"

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {tuple(PRECISIONS)}, got {self.precision!r}")
        if not self.cfl > 0.0:
            raise ValueError(f"cfl must be positive, got {self.cfl}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if not 0.0 < self.tol < 1.0:
            raise ValueError(f"tol must lie in (0, 1), got {self.tol}")
        if self.forces_tol is not None and not self.forces_tol > 0.0:
            raise ValueError(f"forces_tol must be positive or null, got {self.forces_tol}")
        if not 0.0 < self.equation_tol < 1.0 or self.equation_abs_tol <= 0.0:
            raise ValueError("equation tolerances must be positive, with equation_tol below 1")
        if self.n_ghost < 2:
            raise ValueError(
                f"n_ghost must be >= 2 for MUSCL reconstruction, got {self.n_ghost}"
            )
        if not -1.0 <= self.kappa <= 1.0:
            raise ValueError(f"muscl_kappa must lie in [-1, 1], got {self.kappa}")
        if self.kappa != 0.0 and self.limiter != "none":
            raise ValueError("muscl_kappa other than 0 is implemented without a limiter only "
                             f"(limiter 'none'), got limiter {self.limiter!r}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")

    @property
    def dtype(self) -> np.dtype:
        """Field arithmetic; geometry construction stays in double precision."""
        return np.dtype(PRECISIONS[self.precision])


@dataclass(frozen=True)
class ImplicitConfig:
    """Pseudo-transient Newton-Krylov settings (solvers/implicit.py; meanings in
    ``docs/case_reference.md``, reasons in ``docs/dev/notes.md``)."""

    cfl: float = 10.0
    cfl_max: float = 1.0e3
    cfl_growth: float = 1.3
    cfl_cutback: float = 4.0
    cfl_min: float = 0.5
    preconditioner: str = "line_j"
    jacobian: str = "first_order"
    krylov_tol: float = 1.0e-2
    krylov_subspace: int = 10
    krylov_restarts: int = 2
    krylov_success: float = 0.5
    max_relative_update: float | None = None
    nonlinear_growth_limit: float | None = None

    def __post_init__(self) -> None:
        if not self.cfl > 0.0:
            raise ValueError(f"cfl must be positive, got {self.cfl}")
        if self.cfl_max < self.cfl:
            raise ValueError(
                f"cfl_max must be at least the starting cfl {self.cfl}, "
                f"got {self.cfl_max}"
            )
        if not 0.0 < self.cfl_min <= self.cfl:
            raise ValueError(
                f"cfl_min must lie in (0, cfl], got {self.cfl_min}"
            )
        if self.cfl_growth < 1.0:
            raise ValueError(f"cfl_growth must be >= 1, got {self.cfl_growth}")
        if self.cfl_cutback <= 1.0:
            raise ValueError(f"cfl_cutback must exceed 1, got {self.cfl_cutback}")
        if self.jacobian not in ("exact", "first_order"):
            raise ValueError(
                f"jacobian must be 'exact' or 'first_order', got {self.jacobian!r}"
            )
        if not 0.0 < self.krylov_tol < 1.0:
            raise ValueError(f"krylov_tol must lie in (0, 1), got {self.krylov_tol}")
        if self.krylov_subspace < 1:
            raise ValueError(
                f"krylov_subspace must be >= 1, got {self.krylov_subspace}"
            )
        if self.krylov_restarts < 1:
            raise ValueError(
                f"krylov_restarts must be >= 1, got {self.krylov_restarts}"
            )
        if self.max_relative_update is not None and not self.max_relative_update > 0.0:
            raise ValueError(
                f"max_relative_update must be positive or None, got {self.max_relative_update}"
            )
        if self.nonlinear_growth_limit is not None and self.nonlinear_growth_limit < 1.0:
            raise ValueError("nonlinear_growth_limit must be at least 1 or None")
        if not 0.0 < self.krylov_success < 1.0:
            raise ValueError(
                f"krylov_success must lie in (0, 1), got {self.krylov_success}"
            )
