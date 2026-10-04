from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import write_jsonl


def load_dataset_rows(
    *,
    dataset: str,
    split: str,
    name: str | None = None,
    data_files: str | None = None,
) -> list[dict[str, Any]]:
    if dataset in {"json", "jsonl"}:
        return _load_json_rows(Path(data_files or ""))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Loading Hugging Face datasets requires `datasets`.") from exc

    kwargs: dict[str, Any] = {"split": split}
    if name:
        kwargs["name"] = name
    if data_files:
        kwargs["data_files"] = data_files
    hf_dataset = load_dataset(dataset, **kwargs)
    return [dict(row) for row in hf_dataset]


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("data", "rows", "train"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL rows.")
    return [dict(row) for row in data]


def add_index(rows: list[dict[str, Any]], *, id_field: str = "index") -> list[dict[str, Any]]:
    indexed = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault(id_field, idx)
        indexed.append(item)
    return indexed


def split_rows(
    rows: list[dict[str, Any]],
    *,
    easy_ids: list[int],
    hard_ids: list[int],
    retain_rows: list[dict[str, Any]] | None = None,
    id_field: str = "index",
) -> dict[str, list[dict[str, Any]]]:
    rows = add_index(rows, id_field=id_field)
    retain_rows = add_index(retain_rows or [], id_field=id_field)
    by_id = {int(row[id_field]): row for row in rows}

    missing_easy = sorted(set(easy_ids) - set(by_id))
    missing_hard = sorted(set(hard_ids) - set(by_id))
    if missing_easy or missing_hard:
        raise ValueError(
            "Selection contains IDs not present in the dataset: "
            f"easy={missing_easy[:10]}, hard={missing_hard[:10]}"
        )

    easy_set = set(easy_ids)
    hard_set = set(hard_ids)
    forget_easy = [by_id[idx] for idx in easy_ids]
    forget_hard = [by_id[idx] for idx in hard_ids]
    retain_easy = retain_rows + [row for row in rows if int(row[id_field]) not in easy_set]
    retain_hard = retain_rows + [row for row in rows if int(row[id_field]) not in hard_set]

    return {
        "forget_easy": forget_easy,
        "forget_hard": forget_hard,
        "retain_easy": retain_easy,
        "retain_hard": retain_hard,
    }


def write_split(output_dir: str | Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {}, "counts": {}}
    for name, rows in splits.items():
        file_name = f"{name}.jsonl"
        write_jsonl(output / file_name, rows)
        manifest["files"][name] = file_name
        manifest["counts"][name] = len(rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
