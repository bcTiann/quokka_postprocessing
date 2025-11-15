"""Public entry points for DESPOTIC table utilities."""

from .models import (
    LogGrid,
    LineLumResult,
    SpeciesLineGrid,
    AttemptRecord,
    DespoticTable,
)
from .builder import build_table, save_table, plot_table
from .diagnostics import plot_failure_overlay, summarize_failures

__all__ = [
    "LogGrid",
    "LineLumResult",
    "SpeciesLineGrid",
    "AttemptRecord",
    "DespoticTable",
    "build_table",
    "save_table",
    "plot_table",
    "plot_failure_overlay",
    "summarize_failures",
]
