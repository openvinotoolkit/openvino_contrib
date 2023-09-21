import argparse
import logging
import sys


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default="8000")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=str, required=False, default=None)
    parser.add_argument("--device", type=str, required=False, default="CPU")
    parser.add_argument("--assistant", type=str, required=False, default=None)
    parser.add_argument("--summarization-endpoint", action="store_true")

    return parser


def setup_logger():
    logging.setLoggerClass(ServerLogger)
    _set_uvicorn_log_format(ServerLogger.default_formatter._fmt)


def get_logger(
    name: str,
    level: int = logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


class ServerLogger(logging.Logger):
    _server_log_prefix = "[OpenVINO Code Server Log]"

    default_formatter = logging.Formatter(f"{_server_log_prefix} %(asctime)s %(levelname)s %(message)s")

    def __init__(self, name):
        super(ServerLogger, self).__init__(name)

        self.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.WARNING)
        stdout_handler.setFormatter(self.default_formatter)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(self.default_formatter)

        self.addHandler(stdout_handler)
        self.addHandler(stderr_handler)


def _set_uvicorn_log_format(format: str):
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["access"]["fmt"] = format
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = format
