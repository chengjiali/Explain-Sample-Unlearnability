#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cud_difficulty.tofu_split import load_dataset_rows, split_rows, write_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split TOFU rows into easy/hard JSONL files.")
    parser.add_argument("--selection", required=True, help="selected_samples.json from selection step.")
    parser.add_argument("--dataset", default="locuslab/TOFU")
    parser.add_argument("--name", default="forget10")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--retain-dataset", default=None)
    parser.add_argument("--retain-name", default=None)
    parser.add_argument("--retain-split", default="train")
    parser.add_argument("--retain-data-files", default=None)
    parser.add_argument("--id-field", default="index")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = json.loads(Path(args.selection).read_text())
    rows = load_dataset_rows(
        dataset=args.dataset,
        name=args.name,
        split=args.split,
        data_files=args.data_files,
    )
    retain_rows = None
    if args.retain_dataset:
        retain_rows = load_dataset_rows(
            dataset=args.retain_dataset,
            name=args.retain_name,
            split=args.retain_split,
            data_files=args.retain_data_files,
        )
    splits = split_rows(
        rows,
        easy_ids=[int(i) for i in selection["easy"]],
        hard_ids=[int(i) for i in selection["hard"]],
        retain_rows=retain_rows,
        id_field=args.id_field,
    )
    write_split(args.output_dir, splits)
    print(f"wrote TOFU easy/hard splits to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
