"""Public entry points for DESPOTIC table utilities."""

from .models import (
    LogGrid,
    LineLumResult,
    SpeciesLineGrid,
    AttemptRecord,
    DespoticTable,
)
from .builder import build_table, plot_table
from .io import load_table, save_table
from .diagnostics import plot_failure_overlay, summarize_failures, plot_sampling_histogram
from .plotting import plot_table_overview

__all__ = [
    "LogGrid",
    "LineLumResult",
    "SpeciesLineGrid",
    "AttemptRecord",
    "DespoticTable",
    "build_table",
    "save_table",
    "load_table",
    "plot_table",
    "plot_failure_overlay",
    "summarize_failures",
    "plot_sampling_histogram",
    "plot_table_overview",
]
