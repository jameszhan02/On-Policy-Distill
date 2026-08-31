#!/usr/bin/env python3
"""Create a tiny parquet dataset for OPD smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXAMPLES = [
    ("What is 1+1? Give the final answer in the form #### number.", "2"),
    ("Solve: 3 + 5. Give the final answer in the form #### number.", "8"),
    ("There are 4 apples and 2 more are added. How many apples are there? Give the final answer in the form #### number.", "6"),
    ("A box has 10 pencils. If 3 are removed, how many remain? Give the final answer in the form #### number.", "7"),
    ("What is 2 times 6? Give the final answer in the form #### number.", "12"),
    ("What is 15 minus 9? Give the final answer in the form #### number.", "6"),
    ("If each bag has 3 cookies and there are 4 bags, how many cookies are there? Give the final answer in the form #### number.", "12"),
    ("Complete the sequence 2, 4, 6, 8. What is the next number? Give the final answer in the form #### number.", "10"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/smoke", help="Directory for train.parquet and val.parquet.")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat the prompt list this many times.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (prompt, answer) in enumerate(EXAMPLES * args.repeat):
        rows.append(
            {
                # verl's RLHFDataset._build_messages() does `messages = example.pop(prompt_key)`
                # and expects a list of chat-message dicts (see examples/data_preprocess/gsm8k.py),
                # not a bare string - apply_chat_template() silently drops a bare string's content
                # instead of raising, which used to leave validation prompts with no user turn.
                "prompt": [{"role": "user", "content": prompt}],
                "data_source": "openai/gsm8k",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": "smoke", "index": idx},
            }
        )

    df = pd.DataFrame(rows)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    df.to_parquet(train_path)
    df.to_parquet(val_path)

    print(f"wrote {len(df)} rows")
    print(f"train: {train_path}")
    print(f"val:   {val_path}")


if __name__ == "__main__":
    main()
