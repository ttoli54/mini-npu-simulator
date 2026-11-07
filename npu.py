"""Core calculations for the Mini NPU simulator."""

from __future__ import annotations

import math
from typing import Sequence


EPSILON = 1e-9


def mac_2d(pattern: Sequence[Sequence[float]], filter_matrix: Sequence[Sequence[float]]) -> float:
    """Return the sum of element-wise products for two equally shaped matrices."""
    if len(pattern) != len(filter_matrix):
        raise ValueError("행렬 크기가 서로 다릅니다.")
    if not pattern:
        return 0.0
    if any(len(row) != len(filter_matrix[index]) for index, row in enumerate(pattern)):
        raise ValueError("행렬 크기가 서로 다릅니다.")

    score = 0.0
    for row_index, row in enumerate(pattern):
        for column_index, value in enumerate(row):
            score += value * filter_matrix[row_index][column_index]
    return score


def mac_1d(pattern: Sequence[float], filter_values: Sequence[float]) -> float:
    """Return the sum of element-wise products for two flattened matrices."""
    if len(pattern) != len(filter_values):
        raise ValueError("1차원 배열 크기가 서로 다릅니다.")
    size = math.isqrt(len(pattern))
    if size * size != len(pattern):
        raise ValueError("1차원 배열 길이는 정사각형이어야 합니다.")
    return sum(pattern[index] * filter_values[index] for index in range(len(pattern)))


def classify_scores(
    score_a: float,
    score_b: float,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """Return the higher-score label, or UNDECIDED for an epsilon tie."""
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b
