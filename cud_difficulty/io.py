from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_ids(path: str | Path) -> list[int]:
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("ids", "easy", "hard", "indices"):
                if key in data:
                    data = data[key]
                    break
        return [int(x) for x in data]
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_ids(path: str | Path, ids: list[int]) -> None:
    Path(path).write_text(",".join(str(i) for i in ids) + "\n")


def _load_torch(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"Reading {path} requires torch. Install requirements.txt.") from exc
    return torch.load(path, map_location="cpu", weights_only=True)


def _records_from_array(values: Any, method: str) -> pd.DataFrame:
    array = np.asarray(values, dtype=float).reshape(-1)
    return pd.DataFrame(
        {"method": method, "sample_id": np.arange(len(array), dtype=int), "score": array}
    )


def _records_from_mapping(data: dict[str, Any], method: str) -> pd.DataFrame:
    for key in ("score", "scores", "loss", "losses", "difficulty", "cud"):
        if key in data:
            return _records_from_array(data[key], method)
    if all(str(k).isdigit() for k in data):
        rows = [
            {"method": method, "sample_id": int(k), "score": float(v)}
            for k, v in data.items()
        ]
        return pd.DataFrame(rows)
    raise ValueError(
        "Could not find a score vector in mapping. Expected one of "
        "score/scores/loss/losses/difficulty/cud or an id->score mapping."
    )


def load_score_file(
    path: str | Path,
    *,
    method: str | None = None,
    id_column: str = "sample_id",
    score_column: str = "score",
) -> pd.DataFrame:
    path = Path(path)
    method_name = method or path.parent.name or path.stem
    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        data = _load_torch(path)
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        if isinstance(data, dict):
            return _records_from_mapping(data, method_name)
        return _records_from_array(data, method_name)

    if suffix == ".csv":
        frame = pd.read_csv(path)
        if score_column not in frame.columns:
            numeric = frame.select_dtypes(include=["number"]).columns.tolist()
            if not numeric:
                raise ValueError(f"No numeric score column found in {path}")
            score_column = numeric[-1]
        if id_column not in frame.columns:
            frame[id_column] = np.arange(len(frame), dtype=int)
        out = frame[[id_column, score_column]].rename(
            columns={id_column: "sample_id", score_column: "score"}
        )
        out.insert(0, "method", method_name)
        return out

    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list) and data and isinstance(data[0], dict):
            frame = pd.DataFrame(data)
            if score_column not in frame.columns:
                raise ValueError(f"{path} does not contain score column {score_column!r}")
            if id_column not in frame.columns:
                frame[id_column] = np.arange(len(frame), dtype=int)
            out = frame[[id_column, score_column]].rename(
                columns={id_column: "sample_id", score_column: "score"}
            )
            out.insert(0, "method", method_name)
            return out
        if isinstance(data, dict):
            return _records_from_mapping(data, method_name)
        return _records_from_array(data, method_name)

    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        if score_column not in frame.columns:
            raise ValueError(f"{path} does not contain score column {score_column!r}")
        if id_column not in frame.columns:
            frame[id_column] = np.arange(len(frame), dtype=int)
        out = frame[[id_column, score_column]].rename(
            columns={id_column: "sample_id", score_column: "score"}
        )
        out.insert(0, "method", method_name)
        return out

    text = path.read_text().strip()
    values = [float(x) for x in text.replace("\n", ",").split(",") if x.strip()]
    return _records_from_array(values, method_name)
