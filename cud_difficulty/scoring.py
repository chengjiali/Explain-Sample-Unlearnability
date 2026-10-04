from __future__ import annotations

import torch


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine similarity after flattening two tensors."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)
    if norm_a < eps or norm_b < eps:
        return torch.tensor(0.0, device=a.device)
    return torch.dot(a, b) / (norm_a * norm_b + eps)


def anchored_unlearnability_score(
    query: torch.Tensor,
    easy_anchor: torch.Tensor,
    hard_anchor: torch.Tensor,
    *,
    eps: float = 1e-12,
    clamp: bool = True,
) -> torch.Tensor:
    """Return a CUD-style score calibrated by easy and hard circuit anchors.

    The construction maps the easy anchor to 0 and the hard anchor to 1:

        score = d(query, easy) / (d(query, easy) + d(query, hard))

    where d is cosine distance.
    """
    easy_distance = (1.0 - cosine_sim(query, easy_anchor, eps=eps)).clamp_min(0.0)
    hard_distance = (1.0 - cosine_sim(query, hard_anchor, eps=eps)).clamp_min(0.0)
    denom = easy_distance + hard_distance
    score = torch.where(
        denom > eps,
        easy_distance / (denom + eps),
        torch.tensor(0.5, device=query.device),
    )
    return score.clamp(0.0, 1.0) if clamp else score


def vectorize_eap_graph(graph) -> torch.Tensor:
    """Vectorize an EAP/TransformerLens graph after pruning.

    This keeps the useful circuit-vector operation from the exploratory code
    while leaving the heavy graph extraction dependencies outside this package.
    """
    mask = graph.real_edge_mask
    in_graph = graph.in_graph.clone()[mask].bool()
    scores = graph.scores.clone()[mask].float()
    return scores * in_graph.float()
