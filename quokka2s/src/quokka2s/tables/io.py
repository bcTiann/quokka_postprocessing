# quokka2s/src/quokka2s/tables/io.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .models import AttemptRecord, DespoticTable, SpeciesLineGrid

TABLE_VERSION = 1
_SPECIES_FIELDS = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust", "abundance")


def _attempts_to_array(attempts: Iterable[AttemptRecord]) -> np.ndarray:
    records = list(attempts)
    arr = np.empty(len(records), dtype([
        ("row_idx", np.int32),
        ("col_idx", np.int32),
        ("nH", float),
        ("colDen", float),
        ("tg_guess", float),
        ("final_Tg", float),
        ("converged", np.bool_),
        ("message", object),
        ("duration", float),
    ]))
    for idx, rec in enumerate(records):
        arr[idx] = (
            rec.row_idx,
            rec.col_idx,
            rec.nH,
            rec.colDen,
            rec.tg_guess,
            rec.final_Tg,
            rec.converged,
            rec.message,
            rec.duration if rec.duration is not None else np.nan,
        )
    return arr


def _attempts_from_array(data: np.ndarray) -> tuple[AttemptRecord, ...]:
    attempts: list[AttemptRecord] = []
    for row in data:
        attempts.append(
            AttemptRecord(
                row_idx=int(row["row_idx"]),
                col_idx=int(row["col_idx"]),
                nH=float(row["nH"]),
                colDen=float(row["colDen"]),
                tg_guess=float(row["tg_guess"]),
                final_Tg=float(row["final_Tg"]),
                converged=bool(row["converged"]),
                message=row["message"] if row["message"] else None,
                duration=None if np.isnan(row["duration"]) else float(row["duration"]),
            )
        )
    return tuple(attempts)



def save_table(table: DespoticTable, path: str | Path) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "version": np.array([TABLE_VERSION], dtype=np.int32),
        "nH_values": np.array(table.nH_values),
        "col_density_values": np.array(table.col_density_values),
        "tg_final": np.array(table.tg_final),
        "failure_mask": np.asarray(table.failure_mask) if table.failure_mask is not None else None,
        "energy_rate": np.asarray(table.energy_rate) if table.energy_rate is not None else None,
        "species_names": np.array(table.species, dtype=object),
        "attempts": _attempts_to_array(table.attempts),
    }
    for species, grid in table.species_data.items():
        for field in _SPECIES_FIELDS:
            payload[f"{species}_{field}"] = getattr(grid, field)
            # e.g., "CO_freq", "C+_intTB", etc.
    
    payload = {key: value for key, value in payload.items() if value is not None} # Remove any None (key, value) pairs
    np.savez_compressed(path, **payload)


def load_table(path: str | Path) -> DespoticTable:
    path = Path(path)
    blob = np.load(path, allow_pickle=True)
    version = int(blob["version"])

    if version != TABLE_VERSION:
        raise ValueError(f"Unsupported DESPOTIC table version: {version} (expected {TABLE_VERSION})")
    
    nH_values = np.array(blob["nH_values"])
    col_density_values = np.array(blob["col_density_values"], dtype=float)
    tg_final = np.array(blob["tg_final"], dtype=float)

    failure_mask = blob.get("failure_mask")
    if failure_mask is not None:
        failure_mask = np.array(failure_mask, dtype=bool)

    energy_rate = blob.get("energy_rate")
    if energy_rate is not None:
        energy_rate = np.array(energy_rate, dtype=float)

    species_names = [str(x) for x in blob["species_names"]]
    species_data: dict[str, SpeciesLineGrid] = {}
    for name in species_names:
        fields: Mapping[str, np.ndarray] = {
            field: np.array(blob[f"{name}: {field}"], dtype=float)
            for field in _SPECIES_FIELDS
        }
        species_data[name] = SpeciesLineGrid(**fields)