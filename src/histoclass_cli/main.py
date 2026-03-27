"""CLI entrypoint for histoclass pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from histoclass import project_root

from .pipeline import PipelineMode, format_result_for_console, run_pipeline_from_paths


def build_parser() -> argparse.ArgumentParser:
    """@brief 构建命令行参数解析器；Build command-line argument parser.

    @return 参数解析器对象；Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="histoclass",
        description="IDC patch classification pipeline runner.",
    )
    parser.add_argument(
        "config_name",
        type=str,
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

    config_path = resolve_config_path(args.config_name)
    mode = PipelineMode(args.mode)
    result = run_pipeline_from_paths(
        config_path=config_path,
        mode=mode,
        checkpoint_path=args.checkpoint,
    )
    print(format_result_for_console(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
