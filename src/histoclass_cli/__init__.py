"""CLI package for histoclass application orchestration."""

from .pipeline import (
    BatchPipelineResult,
    PipelineMode,
    PipelineRequest,
    PipelineResult,
    format_result_for_console,
    run_batch_pipeline,
    run_pipeline,
    run_pipeline_from_paths,
)

__all__ = [
    "BatchPipelineResult",
    "PipelineMode",
    "PipelineRequest",
    "PipelineResult",
    "format_result_for_console",
    "run_batch_pipeline",
    "run_pipeline",
    "run_pipeline_from_paths",
]
