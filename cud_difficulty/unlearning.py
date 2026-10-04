from __future__ import annotations

import os
import subprocess
from pathlib import Path


def build_unlearning_command(
    *,
    open_unlearning_dir: str | Path,
    split_name: str,
    splits_dir: str | Path,
    model: str,
    model_path: str,
    trainer: str,
    task_prefix: str = "tofu",
    extra_overrides: list[str] | None = None,
) -> list[str]:
    splits = Path(splits_dir).resolve()
    forget_file = splits / f"forget_{split_name}.jsonl"
    retain_file = splits / f"retain_{split_name}.jsonl"
    if not forget_file.exists():
        raise FileNotFoundError(forget_file)
    if not retain_file.exists():
        raise FileNotFoundError(retain_file)

    task_name = f"{task_prefix}_{model}_{split_name}_{trainer}"
    command = [
        "python",
        "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default.yaml",
        f"trainer={trainer}",
        f"model={model}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        "data.forget.TOFU_QA_forget.args.hf_args.path=json",
        f"data.forget.TOFU_QA_forget.args.hf_args.data_files={forget_file}",
        "data.forget.TOFU_QA_forget.args.hf_args.split=train",
        "data.forget.TOFU_QA_forget.args.hf_args.name=null",
        "data.retain.TOFU_QA_retain.args.hf_args.path=json",
        f"data.retain.TOFU_QA_retain.args.hf_args.data_files={retain_file}",
        "data.retain.TOFU_QA_retain.args.hf_args.split=train",
        "data.retain.TOFU_QA_retain.args.hf_args.name=null",
    ]
    command.extend(extra_overrides or [])
    return command


def run_commands(
    commands: list[list[str]],
    *,
    cwd: str | Path,
    device: str | None = None,
    dry_run: bool = False,
) -> int:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device

    for command in commands:
        printable = " ".join(command)
        print(printable)
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=Path(cwd), env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0
