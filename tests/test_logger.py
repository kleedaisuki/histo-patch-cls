from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

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


def _wait_capture_contains_both(capsys, out_token: str, err_token: str) -> tuple[str, str]:  # noqa: ANN001
    deadline = time.time() + 1.0
    all_out = ""
    all_err = ""
    while time.time() < deadline:
        captured = capsys.readouterr()
        all_out += captured.out
        all_err += captured.err
        if out_token in all_out and err_token in all_err:
            return all_out, all_err
        time.sleep(0.01)
    return all_out, all_err


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

    out_text, err_text = _wait_capture_contains_both(
        capsys, "stream-info-token", "stream-error-token"
    )
    assert "stream-info-token" in out_text
    assert "stream-error-token" in err_text
    assert "thread=" in out_text
    assert "thread=" in err_text

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

    deadline = time.time() + 1.0
    while time.time() < deadline and not log_file.exists():
        time.sleep(0.01)
    assert log_file.exists()

    content = ""
    while time.time() < deadline:
        content = log_file.read_text(encoding="utf-8")
        if "lazy-create-token" in content:
            break
        time.sleep(0.01)
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
    assert all("thread=" in line for line in matched)

    _restore_default_logging()
