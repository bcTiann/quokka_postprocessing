""" Dataclasses for DESPOTIC lookup tables."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"DESPOTIC.*emitterData",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"DESPOTIC.*NL99_GC",
)

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LogGrid:
    """Defines a logarithmic grid."""

    min_value: float
    max_value: float
    num_points: int

    def __post_init__(self) -> None:
        if self.min_value <= 0 or self.max_value <= 0:
            raise ValueError("LogGrid min_value and max_value must be positive.")
        if self.min_value >= self.max_value:
            raise ValueError("LogGrid min_value must be less than max_value.")
        if self.num_points < 2:
            raise ValueError("LogGrid num_points must be at least 2.")
    

    def sample(self) -> np.ndarray:
        """Generate sample points on the logarithmic grid."""
        values = np.logspace(
            np.log10(self.min_value), np.log10(self.max_value), self.num_points
        )
        return values
    

@dataclass(frozen=True)
class LineLumResult:
    """
    Line Luminosity output from a single species.

    Attributes:

        freq: Line frequency in Hz.

        'intIntensity' : float
              frequency-integrated intensity of the line, with the CMB
              contribution subtracted off; units are erg cm^-2 s^-1 sr^-1 

        'intTB' : float
              velocity-integrated brightness temperature of the line,
              with the CMB contribution subtracted off; units are K km
              s^-1

        'lumPerH' : float
              luminosity of the line per H nucleus; units are erg s^-1
              H^-1

        'tau' : float
              optical depth in the line, not including dust

        'tauDust' : float
              dust optical depth in the line
    """

    freq: float
    intIntensity: float
    intTB: float
    lumPerH: float
    tau: float
    tauDust: float

@dataclass(frozen=True)
class SpeciesLineGrid:
    """
    Grid of lineLum outputs for an emitting species.
    """

    freq: np.ndarray
    intIntensity: np.ndarray
    intTB: np.ndarray
    lumPerH: np.ndarray
    tau: np.ndarray
    tauDust: np.ndarray
    abundance: np.ndarray     


@dataclass(frozen=True)
class AttemptRecord:
    """
    Record of a single DESPOTIC attempt
    """

    row_idx: int
    col_idx: int
    nH: float
    colDen: float
    tg_guess: float
    final_Tg: float
    converged: bool
    message: str | None = None
    duration: float | None = None


@dataclass(frozen=True)
class DespoticTable:
    """
    DESPOTIC lookup table for a single species.

    Attributes:

    """

    species_data: Mapping[str, SpeciesLineGrid]
    tg_final: np.ndarray
    nH_values: np.ndarray
    col_density_values: np.ndarray
    attempts: Tuple[AttemptRecord, ...] = field(default_factory=tuple)
    failure_mask: np.ndarray | None = None
    energy_terms: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_data", MappingProxyType(self.species_data))
        if self.failure_mask is not None:
            if self.failure_mask.shape != self.tg_final.shape:
                raise ValueError("failure_mask shape must match tg_final shape.")
        if self.energy_terms is not None:
            object.__setattr__(self, "energy_terms", MappingProxyType(dict(self.energy_terms)))


    @property
    def species(self) -> Tuple[str, ...]:
        return tuple(self.species_data.keys())
    
    def require_species(self, species: str) -> tuple[str, SpeciesLineGrid]:
        if not self.species:
            raise ValueError("DespoticTable contains no species data.")
        if species not in self.species_data:
            available = ", ".join(self.species_data.keys())
            raise ValueError(f"Species '{species}' not found in DespoticTable. Available species: {available}.")
        return species, self.species_data[species]
    
    def clone_species_fields(self) -> dict[str, dict[str, np.ndarray]]:
        field_map: dict[str, dict[str, np.ndarray]] = {}
        for name, grid in self.species_data.items():
            field_map[name] = {
                field: np.array(getattr(grid, field), copy=True)
                for field in ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
            }
        return field_map
    

