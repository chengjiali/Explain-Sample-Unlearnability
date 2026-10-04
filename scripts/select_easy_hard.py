#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cud_difficulty.selection import (
    load_scores,
    save_selection,
    select_easy_hard,
    summarize_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select easy/hard sample IDs from scores.")
    parser.add_argument("--score-path", action="append", default=[], help="Score file path.")
    parser.add_argument("--score-glob", action="append", default=[], help="Glob for score files.")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--min-appearances", type=int, default=None)
    parser.add_argument(
        "--easy",
        choices=["low", "high"],
        default="low",
        help="Whether low or high scores indicate easy-to-unlearn samples.",
    )
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.score_path)
    for pattern in args.score_glob:
        paths.extend(sorted(glob(pattern)))
    if not paths:
        raise SystemExit("No score files were provided.")

    scores = load_scores(paths, id_column=args.id_column, score_column=args.score_column)
    easy_ids, hard_ids, ranked = select_easy_hard(
        scores,
        num_samples=args.num_samples,
        easy=args.easy,
        min_appearances=args.min_appearances,
    )
    save_selection(
        args.output_dir,
        easy_ids=easy_ids,
        hard_ids=hard_ids,
        ranked_scores=ranked,
        metadata={
            "score_paths": paths,
            "num_samples": args.num_samples,
            "min_appearances": args.min_appearances,
            "easy": args.easy,
        },
    )
    print(summarize_selection(easy_ids, hard_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
