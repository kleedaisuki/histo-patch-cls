"""Application-level pipeline orchestration for histoclass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import gc
import json
import queue
from pathlib import Path
import sys
import threading
import traceback
from typing import Any, Callable, Sequence

import torch

from histoclass import (
    AppConfig,
    EvaluationResult,
    Evaluator,
    Trainer,
    TrainSummary,
    build_data_module,
    build_model,
    config_to_dict,
    load_config,
)
from histoclass.utils import SeedState, get_logger, seed_everything


LOGGER = get_logger(__name__)


class PipelineMode(str, Enum):
    """@brief Pipeline 运行模式；Pipeline execution mode."""

    TRAIN = "train"
    EVAL = "eval"
    TRAIN_EVAL = "train_eval"


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """@brief Pipeline 请求参数；Pipeline request parameters.

    @param config_path 配置文件路径，None 表示使用默认配置；Config path, None to use default config.
    @param mode 运行模式；Execution mode.
    @param checkpoint_path 可选 checkpoint 路径；Optional checkpoint path.
    """

    config_path: str | Path | None = None
    mode: PipelineMode = PipelineMode.TRAIN_EVAL
    checkpoint_path: str | Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """@brief Pipeline 执行结果；Pipeline execution result.

    @param config 生效配置；Resolved application config.
    @param seed_state 随机种子状态；Applied seed state.
    @param train_summary 训练结果；Training summary.
    @param evaluation 评估结果；Evaluation result.
    @param checkpoint_path 本次使用或生成的 checkpoint 路径；Used/generated checkpoint path.
    """

    config: AppConfig
    seed_state: SeedState
    train_summary: TrainSummary | None
    evaluation: EvaluationResult | None
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class BatchPipelineResult:
    """@brief 批量 pipeline 结果；Batch pipeline execution result.

    @param results 每个请求对应的执行结果；Per-request execution results.
    """

    results: tuple[PipelineResult, ...]


@dataclass(slots=True)
class _PreparedRequest:
    """@brief 预处理后的请求上下文；Prepared request runtime context."""

    index: int
    request: PipelineRequest
    config: AppConfig
    seed_state: SeedState
    model: torch.nn.Module
    checkpoint_path: Path | None
    train_summary: TrainSummary | None = None
    evaluation: EvaluationResult | None = None


@dataclass(frozen=True, slots=True)
class _BroadcastJob:
    """@brief 广播任务描述；Broadcast job descriptor.

    @param name 任务名称；Job name.
    @param fn 消费者工作函数；Consumer worker function.
    @param passes 需要消费 source_loader 的轮次；Number of source-loader passes required.
    """

    name: str
    fn: Callable[[Any], None]
    passes: int = 1


class _QueueIterable:
    """@brief 队列可迭代包装器；Queue-backed iterable wrapper."""

    def __init__(
        self,
        data_queue: "queue.Queue[object]",
        *,
        stop_token: object,
        epoch_token: object,
    ) -> None:
        self._data_queue = data_queue
        self._stop_token = stop_token
        self._epoch_token = epoch_token
        self._closed = False

    def __iter__(self):  # noqa: ANN204
        if self._closed:
            return
        while True:
            item = self._data_queue.get()
            if item is self._stop_token:
                self._closed = True
                return
            if item is self._epoch_token:
                return
            yield item


def _signal_queue_token(
    data_queue: "queue.Queue[object]",
    token: object,
    *,
    stage_name: str,
    token_name: str,
    queue_name: str,
) -> None:
    """@brief 无阻塞注入控制标记；Inject control token without deadlock.

    @param data_queue 目标队列；Target queue.
    @param token 控制标记对象；Control token object.
    @param stage_name 广播阶段名称；Broadcast stage name.
    @param token_name 控制标记名称；Human-readable control token name.
    @param queue_name 队列名称；Human-readable queue name.
    @note
      当消费者已退出时，队列可能长期满载。本函数会在必要时丢弃旧数据以保证 stop token 可写入，
      避免 finally 阶段死循环。
    """
    dropped_batches = 0
    while True:
        try:
            data_queue.put(token, timeout=0.1)
            if dropped_batches > 0:
                LOGGER.warning(
                    "Broadcast %s injected %s into %s after dropping %d queued batch(es). "
                    "This indicates lossy approximate broadcast under backpressure.",
                    stage_name,
                    token_name,
                    queue_name,
                    dropped_batches,
                )
            return
        except queue.Full:
            try:
                data_queue.get_nowait()
                dropped_batches += 1
            except queue.Empty:
                continue


def _build_batch_group_key(item: _PreparedRequest) -> tuple[Any, int, bool, bool]:
    """@brief 构建批量数据模块缓存键；Build batch data-module cache key.

    @param item 预处理后的请求项；Prepared request item.
    @return 缓存键元组；Cache key tuple.
    @note
      DataLoader 的随机行为依赖 seed / deterministic / benchmark 设置，
      因而这些字段必须纳入共享 DataModule 的缓存键。
    """
    return (
        item.config.data,
        item.seed_state.seed,
        item.seed_state.deterministic,
        item.seed_state.benchmark,
    )


def _format_thread_stack(thread: threading.Thread) -> str:
    """@brief 提取线程栈文本；Extract a formatted Python stack for one thread.

    @param thread 目标线程；Target thread.
    @return 栈文本；Formatted stack text.
    """
    if thread.ident is None:
        return "<thread ident unavailable>"

    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return "<python frame unavailable>"
    return "".join(traceback.format_stack(frame))


def _release_loader_resources(loader: Any, *, loader_name: str) -> None:
    """@brief 显式释放 DataLoader worker 资源；Explicitly release DataLoader worker resources.

    @param loader 待释放的 DataLoader 或兼容对象；Target DataLoader-like object.
    @param loader_name 日志标识名；Human-readable loader name for logs.
    @note
      对于 persistent_workers=True 的 DataLoader，内部 iterator 可能长期持有 worker 进程。
      本函数会尝试调用私有 shutdown 接口并清空 `_iterator`，以便提前释放训练阶段资源。
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:  # pragma: no cover - defensive cleanup path
            LOGGER.exception("Failed to shutdown workers for %s", loader_name)

    try:
        setattr(loader, "_iterator", None)
    except Exception:  # pragma: no cover - defensive cleanup path
        LOGGER.debug("Failed to clear DataLoader iterator handle for %s", loader_name)


def run_pipeline(request: PipelineRequest) -> PipelineResult:
    """@brief 运行应用层 pipeline；Run application-level pipeline.

    @param request pipeline 请求；Pipeline request.
    @return pipeline 执行结果；Pipeline result.
    """
    config = load_config(request.config_path)
    LOGGER.info(
        "Pipeline started: mode=%s, config_path=%s",
        request.mode.value,
        request.config_path,
    )

    seed_state = seed_everything(
        seed=config.seed.seed,
        deterministic=config.seed.deterministic,
        benchmark=config.seed.benchmark,
    )

    data_module = build_data_module(config.data)
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    model = build_model(config.model)

    checkpoint_path = _resolve_checkpoint_path(
        request.checkpoint_path,
        mode=request.mode,
    )
    if checkpoint_path is not None:
        _load_model_checkpoint(model=model, checkpoint_path=checkpoint_path)

    train_summary: TrainSummary | None = None
    evaluation: EvaluationResult | None = None

    try:
        if request.mode in (PipelineMode.TRAIN, PipelineMode.TRAIN_EVAL):
            trainer = Trainer(model=model, config=config.trainer)
            train_summary = trainer.fit(train_loader)
            if train_summary.final_checkpoint is not None:
                checkpoint_path = train_summary.final_checkpoint

        if request.mode in (PipelineMode.EVAL, PipelineMode.TRAIN_EVAL):
            if request.mode == PipelineMode.TRAIN_EVAL:
                _release_loader_resources(train_loader, loader_name="run_pipeline.train_loader")
                train_loader = None
                del data_module
                gc.collect()

            evaluator = Evaluator(model=model, config=config.evaluator)
            evaluation = evaluator.evaluate(val_loader)
    finally:
        if train_loader is not None:
            _release_loader_resources(train_loader, loader_name="run_pipeline.train_loader")
        _release_loader_resources(val_loader, loader_name="run_pipeline.val_loader")

    _log_pipeline_summary(
        mode=request.mode,
        train_summary=train_summary,
        evaluation=evaluation,
        checkpoint_path=checkpoint_path,
    )
    return PipelineResult(
        config=config,
        seed_state=seed_state,
        train_summary=train_summary,
        evaluation=evaluation,
        checkpoint_path=checkpoint_path,
    )


def run_batch_pipeline(requests: Sequence[PipelineRequest]) -> BatchPipelineResult:
    """@brief 广播并发执行多请求；Run batched pipelines with broadcasted concurrent stages.

    @param requests pipeline 请求序列；Pipeline request sequence.
    @return 批量执行结果；Batch execution result.
    @note
      同一 DataModuleConfig 的请求会共享一份 DataLoader 作为生产者（producer），
      并将每个 batch 广播到多个训练/评估线程（consumers）并发执行。
    """
    if not requests:
        raise ValueError("requests must not be empty.")

    prepared: list[_PreparedRequest] = []
    for index, request in enumerate(requests):
        config = load_config(request.config_path)
        LOGGER.info(
            "Batch pipeline item started: mode=%s, config_path=%s",
            request.mode.value,
            request.config_path,
        )

        seed_state = seed_everything(
            seed=config.seed.seed,
            deterministic=config.seed.deterministic,
            benchmark=config.seed.benchmark,
        )
        model = build_model(config.model)

        checkpoint_path = _resolve_checkpoint_path(
            request.checkpoint_path,
            mode=request.mode,
        )
        if checkpoint_path is not None:
            _load_model_checkpoint(model=model, checkpoint_path=checkpoint_path)

        prepared.append(
            _PreparedRequest(
                index=index,
                request=request,
                config=config,
                seed_state=seed_state,
                model=model,
                checkpoint_path=checkpoint_path,
            )
        )

    data_module_cache: dict[Any, Any] = {}
    grouped: dict[Any, list[_PreparedRequest]] = {}
    for item in prepared:
        grouped.setdefault(_build_batch_group_key(item), []).append(item)

    for cache_key, group_items in grouped.items():
        cached_module = data_module_cache.get(cache_key)
        if cached_module is None:
            seed_everything(
                seed=group_items[0].seed_state.seed,
                deterministic=group_items[0].seed_state.deterministic,
                benchmark=group_items[0].seed_state.benchmark,
            )
            data_module = build_data_module(group_items[0].config.data)
            data_module_cache[cache_key] = data_module
            LOGGER.info(
                "Batch data module cache miss: built new module for key=%s", cache_key
            )
        else:
            data_module = cached_module
            LOGGER.info(
                "Batch data module cache hit: reusing existing module for key=%s",
                cache_key,
            )

        train_source_loader = data_module.train_loader
        val_source_loader = data_module.val_loader

        try:
            train_items = [
                item
                for item in group_items
                if item.request.mode in (PipelineMode.TRAIN, PipelineMode.TRAIN_EVAL)
            ]
            if train_items:
                train_jobs = [
                    _BroadcastJob(
                        name=f"train-worker-{item.index}",
                        fn=_build_train_worker(item),
                        passes=item.config.trainer.epochs,
                    )
                    for item in train_items
                ]
                _run_broadcast_stage(
                    stage_name="train",
                    source_loader=train_source_loader,
                    jobs=train_jobs,
                )
                for item in train_items:
                    if (
                        item.train_summary is not None
                        and item.train_summary.final_checkpoint is not None
                    ):
                        item.checkpoint_path = item.train_summary.final_checkpoint

            eval_items = [
                item
                for item in group_items
                if item.request.mode in (PipelineMode.EVAL, PipelineMode.TRAIN_EVAL)
            ]
            if eval_items:
                _release_loader_resources(
                    train_source_loader,
                    loader_name=f"run_batch_pipeline.train_loader[{cache_key}]",
                )
                train_source_loader = None
                data_module_cache.pop(cache_key, None)
                del data_module
                gc.collect()

                eval_jobs = [
                    _BroadcastJob(
                        name=f"eval-worker-{item.index}",
                        fn=_build_eval_worker(item),
                        passes=1,
                    )
                    for item in eval_items
                ]
                _run_broadcast_stage(
                    stage_name="eval",
                    source_loader=val_source_loader,
                    jobs=eval_jobs,
                )
        finally:
            if train_source_loader is not None:
                _release_loader_resources(
                    train_source_loader,
                    loader_name=f"run_batch_pipeline.train_loader[{cache_key}]",
                )
            _release_loader_resources(
                val_source_loader,
                loader_name=f"run_batch_pipeline.val_loader[{cache_key}]",
            )
            data_module_cache.pop(cache_key, None)

    ordered = sorted(prepared, key=lambda item: item.index)
    results: list[PipelineResult] = []
    for item in ordered:
        _log_pipeline_summary(
            mode=item.request.mode,
            train_summary=item.train_summary,
            evaluation=item.evaluation,
            checkpoint_path=item.checkpoint_path,
        )
        results.append(
            PipelineResult(
                config=item.config,
                seed_state=item.seed_state,
                train_summary=item.train_summary,
                evaluation=item.evaluation,
                checkpoint_path=item.checkpoint_path,
            )
        )

    return BatchPipelineResult(results=tuple(results))


def run_pipeline_from_paths(
    *,
    config_path: str | Path | None,
    mode: PipelineMode,
    checkpoint_path: str | Path | None,
) -> PipelineResult:
    """@brief 面向 main.py 的便捷入口；Convenient entrypoint for main.py.

    @param config_path 配置路径；Config path.
    @param mode 运行模式；Execution mode.
    @param checkpoint_path checkpoint 路径；Checkpoint path.
    @return pipeline 执行结果；Pipeline result.
    """
    request = PipelineRequest(
        config_path=config_path,
        mode=mode,
        checkpoint_path=checkpoint_path,
    )
    return run_pipeline(request)


def format_result_for_console(result: PipelineResult) -> str:
    """@brief 将 pipeline 结果序列化为终端文本；Serialize pipeline result to console text.

    @param result pipeline 执行结果；Pipeline result.
    @return 可打印文本；Printable text.
    """
    payload: dict[str, Any] = {
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
    }

    if result.train_summary is not None:
        payload["train"] = {
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

    if result.evaluation is not None:
        payload["eval"] = {
            "samples": result.evaluation.samples,
            "steps": result.evaluation.steps,
            "loss": result.evaluation.loss,
            "metrics": result.evaluation.metrics.to_dict(),
        }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_train_worker(item: _PreparedRequest) -> Callable[[Any], None]:
    """@brief 构建训练工作函数；Build one train worker callable."""
    trainer = Trainer(model=item.model, config=item.config.trainer)

    def _work(loader_iterable: Any) -> None:
        item.train_summary = trainer.fit(loader_iterable)

    return _work


def _build_eval_worker(item: _PreparedRequest) -> Callable[[Any], None]:
    """@brief 构建评估工作函数；Build one eval worker callable."""
    evaluator = Evaluator(model=item.model, config=item.config.evaluator)

    def _work(loader_iterable: Any) -> None:
        item.evaluation = evaluator.evaluate(loader_iterable)

    return _work


def _run_broadcast_stage(
    *,
    stage_name: str,
    source_loader: Any,
    jobs: Sequence[_BroadcastJob | tuple[str, Callable[[Any], None]]],
) -> None:
    """@brief 广播执行阶段；Run one broadcast stage with producer-consumer threads.

    @param stage_name 阶段名称；Stage name.
    @param source_loader 生产者输入迭代器；Producer source iterable.
    @param jobs 消费者任务集合；Consumer job collection.
    """
    if not jobs:
        return

    normalized_jobs: list[_BroadcastJob] = []
    for job in jobs:
        if isinstance(job, _BroadcastJob):
            normalized_jobs.append(job)
            continue
        if len(job) != 2:
            raise ValueError("jobs tuple format must be (name, fn).")
        normalized_jobs.append(_BroadcastJob(name=job[0], fn=job[1], passes=1))

    for job in normalized_jobs:
        if job.passes <= 0:
            raise ValueError("job passes must be a positive integer.")

    stop_token = object()
    epoch_token = object()
    data_queues = [queue.Queue(maxsize=8) for _ in normalized_jobs]
    stop_event = threading.Event()
    errors: list[BaseException] = []
    error_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def _record_error(exc: BaseException) -> None:
        with error_lock:
            if not errors:
                errors.append(exc)
        stop_event.set()

    def _consume(
        *,
        name: str,
        fn: Callable[[Any], None],
        data_queue: "queue.Queue[object]",
    ) -> None:
        try:
            fn(
                _QueueIterable(
                    data_queue,
                    stop_token=stop_token,
                    epoch_token=epoch_token,
                )
            )
        except BaseException as exc:  # pragma: no cover - exceptional path
            LOGGER.exception("Broadcast %s consumer failed: %s", stage_name, name)
            _record_error(exc)

    for idx, job in enumerate(normalized_jobs):
        thread = threading.Thread(
            target=_consume,
            kwargs={"name": job.name, "fn": job.fn, "data_queue": data_queues[idx]},
            name=f"{stage_name}-{idx}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    try:
        max_passes = max(job.passes for job in normalized_jobs)
        for pass_idx in range(max_passes):
            if stop_event.is_set():
                break

            active_indexes = [
                idx
                for idx, job in enumerate(normalized_jobs)
                if pass_idx < job.passes and threads[idx].is_alive()
            ]
            if not active_indexes:
                break

            for batch in source_loader:
                if stop_event.is_set():
                    break
                for idx in active_indexes:
                    if not threads[idx].is_alive():
                        continue
                    data_queue = data_queues[idx]
                    while not stop_event.is_set():
                        try:
                            data_queue.put(batch, timeout=0.1)
                            break
                        except queue.Full:
                            if not threads[idx].is_alive():
                                break
                            continue

            for idx in active_indexes:
                if threads[idx].is_alive():
                    _signal_queue_token(
                        data_queues[idx],
                        epoch_token,
                        stage_name=stage_name,
                        token_name="epoch_token",
                        queue_name=normalized_jobs[idx].name,
                    )
    except BaseException as exc:  # pragma: no cover - exceptional path
        _record_error(exc)
    finally:
        stop_event.set()
        for idx, data_queue in enumerate(data_queues):
            _signal_queue_token(
                data_queue,
                stop_token,
                stage_name=stage_name,
                token_name="stop_token",
                queue_name=normalized_jobs[idx].name,
            )
        for thread in threads:
            thread.join(timeout=30.0)
            if thread.is_alive():
                LOGGER.error(
                    "Broadcast %s thread join timed out: %s\n%s",
                    stage_name,
                    thread.name,
                    _format_thread_stack(thread),
                )
                raise RuntimeError(
                    f"Broadcast stage '{stage_name}' thread did not terminate: {thread.name}"
                )

    if errors:
        raise RuntimeError(f"Broadcast stage '{stage_name}' failed.") from errors[0]


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    *,
    mode: PipelineMode,
) -> Path | None:
    if checkpoint_path is None:
        if mode == PipelineMode.EVAL:
            raise ValueError("checkpoint_path is required when mode='eval'.")
        return None
    return Path(checkpoint_path).expanduser().resolve()


def _load_model_checkpoint(*, model: torch.nn.Module, checkpoint_path: Path) -> None:
    """@brief 加载模型 checkpoint；Load model checkpoint into model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint payload misses key 'model_state_dict'.")

    model.load_state_dict(state_dict)
    LOGGER.info("Checkpoint loaded: %s", checkpoint_path)


def _log_pipeline_summary(
    *,
    mode: PipelineMode,
    train_summary: TrainSummary | None,
    evaluation: EvaluationResult | None,
    checkpoint_path: Path | None,
) -> None:
    if train_summary is not None:
        final_epoch = train_summary.history[-1]
        LOGGER.info(
            "Pipeline train summary | mode=%s epoch=%d train_loss=%.6f train_f1=%.4f",
            mode.value,
            final_epoch.epoch,
            final_epoch.train.loss,
            final_epoch.train.metrics.f1,
        )

    if evaluation is not None:
        LOGGER.info(
            "Pipeline eval summary | mode=%s val_loss=%s val_f1=%.4f val_auc=%s",
            mode.value,
            (f"{evaluation.loss:.6f}" if evaluation.loss is not None else "None"),
            evaluation.metrics.f1,
            (
                f"{evaluation.metrics.roc_auc:.4f}"
                if evaluation.metrics.roc_auc is not None
                else "None"
            ),
        )

    LOGGER.info(
        "Pipeline completed | mode=%s checkpoint=%s", mode.value, checkpoint_path
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
