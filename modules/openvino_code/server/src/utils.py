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


default_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    formatter: logging.Formatter = default_formatter,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.WARNING)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger
