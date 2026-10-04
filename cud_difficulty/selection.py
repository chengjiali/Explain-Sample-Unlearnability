from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_score_file, write_ids


def load_scores(
    score_paths: list[str | Path],
    *,
    id_column: str = "sample_id",
    score_column: str = "score",
) -> pd.DataFrame:
    frames = [
        load_score_file(path, id_column=id_column, score_column=score_column)
        for path in score_paths
    ]
    if not frames:
        raise ValueError("At least one score path is required.")
    scores = pd.concat(frames, ignore_index=True)
    scores["sample_id"] = scores["sample_id"].astype(int)
    scores["score"] = scores["score"].astype(float)
    return scores


def _rank_frame(scores: pd.DataFrame, easy: str) -> pd.DataFrame:
    if easy not in {"low", "high"}:
        raise ValueError("--easy must be either 'low' or 'high'")

    ascending_for_easy = easy == "low"
    ranked = scores.copy()
    ranked["easy_rank"] = ranked.groupby("method")["score"].rank(
        method="first", ascending=ascending_for_easy
    )
    ranked["hard_rank"] = ranked.groupby("method")["score"].rank(
        method="first", ascending=not ascending_for_easy
    )
    return ranked


def _vote_and_fill(
    ranked: pd.DataFrame,
    *,
    rank_column: str,
    num_samples: int,
    min_appearances: int,
) -> list[int]:
    top = ranked[ranked[rank_column] <= num_samples]
    votes = top.groupby("sample_id").size().rename("votes")
    avg_rank = ranked.groupby("sample_id")[rank_column].mean().rename("avg_rank")
    table = pd.concat([votes, avg_rank], axis=1).fillna({"votes": 0})
    table = table.sort_values(["votes", "avg_rank"], ascending=[False, True])

    selected = table[table["votes"] >= min_appearances].index.astype(int).tolist()
    if len(selected) < num_samples:
        fill = [idx for idx in table.index.astype(int).tolist() if idx not in selected]
        selected.extend(fill[: num_samples - len(selected)])
    return selected[:num_samples]


def select_easy_hard(
    scores: pd.DataFrame,
    *,
    num_samples: int,
    easy: str = "low",
    min_appearances: int | None = None,
) -> tuple[list[int], list[int], pd.DataFrame]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    methods = sorted(scores["method"].unique())
    threshold = min_appearances or max(1, math.ceil(len(methods) / 2))
    ranked = _rank_frame(scores, easy)

    easy_ids = _vote_and_fill(
        ranked,
        rank_column="easy_rank",
        num_samples=num_samples,
        min_appearances=threshold,
    )
    hard_ids = _vote_and_fill(
        ranked,
        rank_column="hard_rank",
        num_samples=num_samples,
        min_appearances=threshold,
    )

    overlap = set(easy_ids) & set(hard_ids)
    if overlap:
        hard_ids = [idx for idx in hard_ids if idx not in overlap]
        hard_fill = (
            ranked.groupby("sample_id")["hard_rank"]
            .mean()
            .sort_values()
            .index.astype(int)
            .tolist()
        )
        hard_ids.extend(
            idx for idx in hard_fill if idx not in set(easy_ids) and idx not in set(hard_ids)
        )
        hard_ids = hard_ids[:num_samples]

    return easy_ids, hard_ids, ranked


def save_selection(
    output_dir: str | Path,
    *,
    easy_ids: list[int],
    hard_ids: list[int],
    ranked_scores: pd.DataFrame,
    metadata: dict,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranked_scores.sort_values(["method", "sample_id"]).to_csv(
        output / "ranked_scores.csv", index=False
    )
    write_ids(output / "easy_ids.txt", easy_ids)
    write_ids(output / "hard_ids.txt", hard_ids)
    payload = {
        "easy": [int(i) for i in easy_ids],
        "hard": [int(i) for i in hard_ids],
        "metadata": metadata,
    }
    (output / "selected_samples.json").write_text(json.dumps(payload, indent=2) + "\n")


def summarize_selection(easy_ids: list[int], hard_ids: list[int]) -> str:
    return (
        f"selected {len(easy_ids)} easy and {len(hard_ids)} hard samples "
        f"({len(set(easy_ids) & set(hard_ids))} overlap)"
    )
