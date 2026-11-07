"""Core calculations for the Mini NPU simulator."""

from __future__ import annotations

import math
from typing import Sequence


EPSILON = 1e-9


def normalize_label(value: object) -> str:
    """Normalize the supported Cross/X aliases to the report labels."""
    if not isinstance(value, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    normalized = value.strip().lower()
    if normalized in {"+", "cross"}:
        return "Cross"
    if normalized == "x":
        return "X"
    raise ValueError(f"지원하지 않는 라벨입니다: {value}")


def validate_matrix(matrix: Sequence[Sequence[float]], expected_size: int | None = None) -> int:
    """Validate a finite square numeric matrix and return its size."""
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError("행렬은 비어 있지 않은 2차원 배열이어야 합니다.")
    size = len(matrix)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"행렬 크기가 {expected_size}x{expected_size}가 아닙니다.")
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != size:
            raise ValueError("행렬은 정사각형이어야 합니다.")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("행렬 값은 숫자여야 합니다.")
            if not math.isfinite(value):
                raise ValueError("NaN과 무한대는 사용할 수 없습니다.")
    return size


def _validate_generated_size(size: int) -> None:
    if not isinstance(size, int) or isinstance(size, bool) or size < 3 or size % 2 == 0:
        raise ValueError("패턴 크기는 3 이상의 홀수여야 합니다.")


def generate_cross(size: int) -> list[list[float]]:
    """Generate an N×N Cross pattern for an odd N."""
    _validate_generated_size(size)
    center = size // 2
    return [
        [1.0 if row == center or column == center else 0.0 for column in range(size)]
        for row in range(size)
    ]


def generate_x(size: int) -> list[list[float]]:
    """Generate an N×N X pattern for an odd N."""
    _validate_generated_size(size)
    return [
        [1.0 if column == row or column == size - row - 1 else 0.0 for column in range(size)]
        for row in range(size)
    ]


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
