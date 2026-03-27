from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from histoclass.utils import configure_logging, get_logger


def _restore_default_logging() -> None:
    configure_logging(
        level="INFO",
        level_targets={
            "DEBUG": "stdout",
            "INFO": "stdout",
            "WARNING": "stderr",
            "ERROR": "stderr",
            "CRITICAL": "stderr",
        },
        file_path=None,
    )


def test_log_stream_routing(capsys) -> None:
    configure_logging(
        level="DEBUG",
        level_targets={
            "DEBUG": "stdout",
            "INFO": "stdout",
            "WARNING": "stderr",
            "ERROR": "stderr",
            "CRITICAL": "stderr",
        },
        file_path=None,
    )
    logger = get_logger("tests.logger.stream")

    logger.info("stream-info-token")
    logger.error("stream-error-token")

    captured = capsys.readouterr()
    assert "stream-info-token" in captured.out
    assert "stream-error-token" in captured.err

    _restore_default_logging()


def test_log_file_is_lazy_created(tmp_path: Path) -> None:
    log_file = tmp_path / "lazy.log"
    configure_logging(
        level="DEBUG",
        level_targets={
            "DEBUG": "stdout",
            "INFO": "file",
            "WARNING": "stderr",
            "ERROR": "stderr",
            "CRITICAL": "stderr",
        },
        file_path=log_file,
    )
    logger = get_logger("tests.logger.file")

    assert not log_file.exists()
    logger.info("lazy-create-token")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "lazy-create-token" in content

    _restore_default_logging()


def test_log_file_concurrent_writes(tmp_path: Path) -> None:
    log_file = tmp_path / "concurrent.log"
    configure_logging(
        level="DEBUG",
        level_targets={
            "DEBUG": "file",
            "INFO": "file",
            "WARNING": "file",
            "ERROR": "file",
            "CRITICAL": "file",
        },
        file_path=log_file,
    )
    logger = get_logger("tests.logger.concurrent")

    def _write_one(i: int) -> None:
        logger.info("concurrent-token-%d", i)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write_one, range(200)))

    lines = log_file.read_text(encoding="utf-8").splitlines()
    matched = [line for line in lines if "concurrent-token-" in line]
    assert len(matched) == 200

    _restore_default_logging()
