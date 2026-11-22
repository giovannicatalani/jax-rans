"""Local wing geometry for sectional calculations.

The zero-taper limit is an infinite swept wing; finite taper uses the
leading- and trailing-edge angle bisector as a fixed Cartesian frame.
"""

from dataclasses import dataclass
from math import atan, degrees, isfinite, radians, tan


@dataclass(frozen=True)
class SectionalSpec:
    """Sweep at the leading edge; ``taper_rate = -dc/ds`` along geometric span."""

    model: str = "conical"
    leading_edge_sweep_deg: float = 0.0
    taper_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.model != "conical":
            raise ValueError(f"sectional model must be 'conical', got {self.model!r}")
        if not isfinite(self.leading_edge_sweep_deg) or not abs(self.leading_edge_sweep_deg) < 90.0:
            raise ValueError("leading_edge_sweep_deg must be finite and lie in (-90, 90)")
        if not isfinite(self.taper_rate):
            raise ValueError("taper_rate must be finite")

    @property
    def trailing_edge_sweep_deg(self) -> float:
        return degrees(atan(tan(radians(self.leading_edge_sweep_deg)) - self.taper_rate))

    @property
    def reference_sweep_deg(self) -> float:
        # Preserve the existing normal frame exactly at zero taper.
        if self.taper_rate == 0.0:
            return self.leading_edge_sweep_deg
        return 0.5 * (self.leading_edge_sweep_deg + self.trailing_edge_sweep_deg)
