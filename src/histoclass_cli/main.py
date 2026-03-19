"""CLI entrypoint for histoclass pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file. Uses default config when omitted.",
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


def main(argv: list[str] | None = None) -> int:
    """@brief CLI 主函数；CLI main function.

    @param argv 可选命令行参数列表；Optional command-line argument list.
    @return 进程退出码；Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    mode = PipelineMode(args.mode)
    result = run_pipeline_from_paths(
        config_path=args.config,
        mode=mode,
        checkpoint_path=args.checkpoint,
    )
    print(format_result_for_console(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
