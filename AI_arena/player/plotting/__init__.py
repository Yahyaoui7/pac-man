"""Plotting and report generation package for training logs."""

from __future__ import annotations

from AI_arena.player.plotting.charts import plot_all
from AI_arena.player.plotting.parser import parse_log
from AI_arena.player.plotting.report import generate_markdown_report

__all__ = ["parse_log", "plot_all", "generate_markdown_report"]
