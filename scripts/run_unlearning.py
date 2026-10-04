#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cud_difficulty.unlearning import build_unlearning_command, run_commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run open-unlearning on easy and hard splits.")
    parser.add_argument("--open-unlearning-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--model", default="Llama-3.2-1B-Instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trainer", default="GradDiff")
    parser.add_argument("--task-prefix", default="tofu")
    parser.add_argument("--split", choices=["easy", "hard"], action="append", default=None)
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_names = args.split or ["easy", "hard"]
    commands = [
        build_unlearning_command(
            open_unlearning_dir=args.open_unlearning_dir,
            split_name=split_name,
            splits_dir=args.splits_dir,
            model=args.model,
            model_path=args.model_path,
            trainer=args.trainer,
            task_prefix=args.task_prefix,
            extra_overrides=args.override,
        )
        for split_name in split_names
    ]
    return run_commands(
        commands,
        cwd=args.open_unlearning_dir,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
