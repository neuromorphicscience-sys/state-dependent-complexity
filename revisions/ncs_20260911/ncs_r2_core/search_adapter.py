from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .shared_objective import select_index


@dataclass
class SearchResult:
    arm: str
    best_mask: tuple[int, ...]
    best_score: float
    history: list[dict]
    candidate_evaluations: int
    state_evus: int
    evolution_used: bool


def random_fixed_k(rng: np.random.Generator, population: int, n: int, k: int) -> np.ndarray:
    if not 0 <= k <= n:
        raise ValueError("invalid fixed k")
    output = np.zeros((population, n), dtype=bool)
    for row in output:
        row[rng.choice(n, size=k, replace=False)] = True
    return output


def _swap(rng: np.random.Generator, masks: np.ndarray, swaps: int) -> np.ndarray:
    children = masks.copy()
    for row in children:
        for _ in range(swaps):
            selected, absent = np.flatnonzero(row), np.flatnonzero(~row)
            row[int(rng.choice(selected))] = False
            row[int(rng.choice(absent))] = True
    return children


def run_search_interface(*, arm: str, n: int, k: int, population: int,
                         generations: int, elite: int, random_injection: int,
                         optimizer_rng_seed: int,
                         score_callback: Callable[[np.ndarray, str, int], np.ndarray],
                         initial_masks: list[tuple[int, ...]] | None = None) -> SearchResult:
    """Runnable policy contract; the callback is the only evaluation boundary."""
    if arm not in ("A_shared", "A_mid", "A_sparse", "shared_random"):
        raise ValueError("unknown arm")
    rng = np.random.default_rng(optimizer_rng_seed)
    masks = random_fixed_k(rng, population, n, k)
    if arm != "shared_random" and initial_masks:
        for index, nodes in enumerate(initial_masks[:population]):
            if len(nodes) != k or len(set(nodes)) != k or any(x < 0 or x >= n for x in nodes):
                raise ValueError("invalid static initializer")
            masks[index] = False
            masks[index, list(nodes)] = True
    best_score, best_mask = -np.inf, None
    history = []
    states = ("transition_mid", "sparse_drive") if arm in ("A_shared", "shared_random") else (
        ("transition_mid",) if arm == "A_mid" else ("sparse_drive",))
    for generation in range(generations):
        values = {state: np.asarray(score_callback(masks, state, generation), dtype=float)
                  for state in states}
        if any(v.shape != (population,) for v in values.values()):
            raise ValueError("score callback returned wrong shape")
        objective = (np.minimum(values["transition_mid"], values["sparse_drive"])
                     if len(states) == 2 else values[states[0]])
        winner = int(np.argmax(objective))
        if float(objective[winner]) > best_score:
            best_score, best_mask = float(objective[winner]), masks[winner].copy()
        history.append({"generation": generation, "best": float(objective[winner]),
                        "mean": float(objective.mean())})
        if arm == "shared_random":
            masks = random_fixed_k(rng, population, n, k)
            continue
        elite_count = min(elite, population)
        elite_masks = masks[np.argsort(objective)[-elite_count:]].copy()
        injection = min(random_injection, population - elite_count)
        offspring = population - elite_count - injection
        if offspring:
            parents = elite_masks[rng.integers(0, elite_count, size=offspring)]
            kids = _swap(rng, parents, 2)
        else:
            kids = elite_masks[:0]
        randoms = random_fixed_k(rng, injection, n, k)
        masks = np.concatenate([elite_masks, kids, randoms], axis=0)
    if best_mask is None:
        raise RuntimeError("empty search")
    return SearchResult(arm, tuple(np.flatnonzero(best_mask).tolist()), best_score,
                        history, population * generations,
                        population * generations * len(states), arm != "shared_random")


def select_candidate(arm: str, candidates: list[dict], mid_values, sparse_values) -> dict:
    return candidates[select_index(arm, mid_values, sparse_values)]


def extract_a_pool(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows if row.get("variant") == "primary" and
                int(row.get("repeat", -1)) == 0 and
                row.get("source_state") in ("transition_mid", "sparse_drive")]
    by_graph: dict[int, set[str]] = {}
    for row in eligible:
        by_graph.setdefault(int(row["graph_seed"]), set()).add(row["source_state"])
    if any(states != {"transition_mid", "sparse_drive"} for states in by_graph.values()):
        raise ValueError("A_pool requires exactly both historical state winners per graph")
    return [dict(row, arm="A_pool", exploratory=True) for row in eligible]
