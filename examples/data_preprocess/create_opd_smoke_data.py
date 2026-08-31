#!/usr/bin/env python3
"""Create a tiny parquet dataset for OPD smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROMPTS = [
    "What is 1+1? Answer briefly.",
    "Solve: 3 + 5 =",
    "Write one sentence about gravity.",
    "Answer briefly: what is water?",
    "What color is the daytime sky on a clear day?",
    "Complete the sequence: 2, 4, 6,",
    "Give one reason people use umbrellas.",
    "Translate to English: bonjour.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/smoke", help="Directory for train.parquet and val.parquet.")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat the prompt list this many times.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"prompt": PROMPTS * args.repeat})
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    df.to_parquet(train_path)
    df.to_parquet(val_path)

    print(f"wrote {len(df)} rows")
    print(f"train: {train_path}")
    print(f"val:   {val_path}")


if __name__ == "__main__":
    main()
