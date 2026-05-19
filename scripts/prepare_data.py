#!/usr/bin/env python3
"""Convert raw corpora to data/processed/*.jsonl."""

from _bootstrap import init

init()

from src.paths import load_config  # noqa: E402

from src.data.corpora import convert_corpora  # noqa: E402


def main() -> None:
    convert_corpora(load_config())


if __name__ == "__main__":
    main()
