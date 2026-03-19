"""CLI package for histoclass application orchestration."""

from .pipeline import (
    PipelineMode,
    PipelineRequest,
    PipelineResult,
    format_result_for_console,
    run_pipeline,
    run_pipeline_from_paths,
)

__all__ = [
    "PipelineMode",
    "PipelineRequest",
    "PipelineResult",
    "format_result_for_console",
    "run_pipeline",
    "run_pipeline_from_paths",
]
