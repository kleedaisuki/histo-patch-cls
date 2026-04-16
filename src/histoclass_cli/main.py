"""CLI entrypoint for histoclass pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histoclass import config_to_dict, project_root

from .pipeline import PipelineMode, PipelineRequest, PipelineResult, run_batch_pipeline


def build_parser() -> argparse.ArgumentParser:
    """@brief 构建命令行参数解析器；Build command-line argument parser.

    @return 参数解析器对象；Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="histoclass",
        description="IDC patch classification pipeline runner.",
    )
    parser.add_argument(
        "config_names",
        type=str,
        nargs="+",
        help="Config name (resolved as configs/<name>.json) or direct JSON path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in PipelineMode],
        default=PipelineMode.TRAIN_EVAL.value,
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (required for eval mode).",
    )
    return parser


def resolve_config_path(config_name: str) -> Path:
    """@brief 解析配置参数；Resolve config CLI argument.

    @param config_name 配置名或 JSON 路径；Config name or JSON path.
    @return 解析后的 JSON 配置绝对路径；Resolved absolute JSON config path.
    """
    candidate = Path(config_name).expanduser()
    is_named_only = not candidate.is_absolute() and len(candidate.parts) == 1

    if candidate.suffix.lower() == ".json":
        if not is_named_only:
            return candidate.resolve()
        direct = candidate.resolve()
        if direct.exists():
            return direct
        return (project_root() / "configs" / candidate.name).resolve()

    if not is_named_only:
        return candidate.with_suffix(".json").resolve()

    direct = candidate.with_suffix(".json").resolve()
    if direct.exists():
        return direct
    return (project_root() / "configs" / f"{candidate.name}.json").resolve()


def main(argv: list[str] | None = None) -> int:
    """@brief CLI 主函数；CLI main function.

    @param argv 可选命令行参数列表；Optional command-line argument list.
    @return 进程退出码；Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    mode = PipelineMode(args.mode)
    requests = tuple(
        PipelineRequest(
            config_path=resolve_config_path(config_name),
            mode=mode,
            checkpoint_path=args.checkpoint,
        )
        for config_name in args.config_names
    )
    batch_result = run_batch_pipeline(requests)
    for index, (request, result) in enumerate(
        zip(requests, batch_result.results, strict=True)
    ):
        if index > 0:
            print()
        print(
            json.dumps(
                _serialize_result_for_console(request=request, result=result),
                ensure_ascii=False,
                indent=4,
            )
        )
    return 0


def _serialize_result_for_console(
    *,
    request: PipelineRequest,
    result: PipelineResult,
) -> dict[str, object]:
    """@brief 序列化单条请求结果；Serialize one request-result pair."""
    return {
        "request": {
            "config_path": (
                str(request.config_path) if request.config_path is not None else None
            ),
            "mode": request.mode.value,
            "checkpoint_path": (
                str(request.checkpoint_path)
                if request.checkpoint_path is not None
                else None
            ),
        },
        "config": config_to_dict(result.config),
        "seed": {
            "seed": result.seed_state.seed,
            "deterministic": result.seed_state.deterministic,
            "benchmark": result.seed_state.benchmark,
            "cuda_available": result.seed_state.cuda_available,
        },
        "checkpoint_path": (
            result.checkpoint_path.as_posix()
            if result.checkpoint_path is not None
            else None
        ),
        "train": (
            {
                "epochs": len(result.train_summary.history),
                "final_checkpoint": (
                    result.train_summary.final_checkpoint.as_posix()
                    if result.train_summary.final_checkpoint is not None
                    else None
                ),
                "final_train_loss": result.train_summary.history[-1].train.loss,
                "final_train_metrics": result.train_summary.history[
                    -1
                ].train.metrics.to_dict(),
            }
            if result.train_summary is not None
            else None
        ),
        "eval": (
            {
                "samples": result.evaluation.samples,
                "steps": result.evaluation.steps,
                "loss": result.evaluation.loss,
                "metrics": result.evaluation.metrics.to_dict(),
            }
            if result.evaluation is not None
            else None
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
