"""Core calculations for the Mini NPU simulator."""

from __future__ import annotations

import math
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


EPSILON = 1e-9
PATTERN_KEY = re.compile(r"^size_(\d+)_\d+$")


@dataclass
class CaseResult:
    case_id: str
    size_key: str = ""
    expected: str = ""
    predicted: str = ""
    status: str = "FAIL"
    reason: str = ""
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(case.status == "PASS" for case in self.cases)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def failures(self) -> List[CaseResult]:
        return [case for case in self.cases if case.status == "FAIL"]


@dataclass
class BenchmarkResult:
    size: int
    average_ms: float
    repeats: int
    operation_count: int


@dataclass
class ComparisonResult:
    size: int
    repeats: int
    score_2d: float
    score_1d: float
    average_2d_ms: float
    average_1d_ms: float


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


def load_data(path: Path) -> dict:
    """Load the original JSON document and reject unreadable root data."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 형식이 올바르지 않습니다: {error.msg}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("filters"), dict) or not isinstance(raw.get("patterns"), dict):
        raise ValueError("JSON에는 filters와 patterns 객체가 필요합니다.")
    return raw


def _failed_case(case_id: str, reason: str, size_key: str = "") -> CaseResult:
    return CaseResult(case_id=case_id, size_key=size_key, reason=reason)


def _normalized_filters(raw_filters: object) -> Dict[str, Sequence[Sequence[float]]]:
    if not isinstance(raw_filters, dict):
        raise ValueError("필터 정보가 객체가 아닙니다.")
    normalized: Dict[str, Sequence[Sequence[float]]] = {}
    for raw_label, matrix in raw_filters.items():
        label = normalize_label(raw_label)
        if label in normalized:
            raise ValueError(f"중복된 필터 라벨입니다: {label}")
        normalized[label] = matrix
    for required in ("Cross", "X"):
        if required not in normalized:
            raise ValueError(f"{required} 필터가 없습니다.")
    return normalized


def analyze_data(data: dict) -> AnalysisReport:
    """Analyze every pattern entry while isolating case-level failures."""
    filters = data.get("filters", {})
    patterns = data.get("patterns", {})
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("filters와 patterns는 객체여야 합니다.")

    report = AnalysisReport()
    for case_id, raw_case in patterns.items():
        case_id = str(case_id)
        match = PATTERN_KEY.fullmatch(case_id)
        if not match:
            report.cases.append(_failed_case(case_id, "패턴 키가 size_N_번호 형식이 아닙니다."))
            continue
        case_result = CaseResult(case_id=case_id)
        try:
            size = int(match.group(1))
            size_key = f"size_{size}"
            case_result.size_key = size_key
            if not isinstance(raw_case, dict):
                raise ValueError("패턴 정보가 객체가 아닙니다.")
            matrices = _normalized_filters(filters.get(size_key))
            pattern = raw_case["input"]
            expected = normalize_label(raw_case["expected"])
            validate_matrix(pattern, size)
            for label, matrix in matrices.items():
                validate_matrix(matrix, size)
            scores = {label: mac_2d(pattern, matrix) for label, matrix in matrices.items()}
            if not all(math.isfinite(score) for score in scores.values()):
                raise ValueError("MAC 결과가 유한하지 않습니다.")
            predicted = classify_scores(scores["Cross"], scores["X"], "Cross", "X")
            case_result.expected = expected
            case_result.predicted = predicted
            case_result.scores = scores
            case_result.status = "PASS" if predicted == expected else "FAIL"
            case_result.reason = "예상 라벨과 일치합니다." if case_result.status == "PASS" else (
                f"판정 {predicted}가 expected {expected}와 다릅니다."
            )
        except (KeyError, TypeError, ValueError) as error:
            case_result.reason = str(error)
        report.cases.append(case_result)
    return report


def _average_mac_time_2d(pattern: Sequence[Sequence[float]], filter_matrix: Sequence[Sequence[float]], repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        mac_2d(pattern, filter_matrix)
    return (time.perf_counter() - start) * 1000 / repeats


def _average_mac_time_1d(pattern: Sequence[float], filter_values: Sequence[float], repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        mac_1d(pattern, filter_values)
    return (time.perf_counter() - start) * 1000 / repeats


def _validate_repeats(repeats: int) -> None:
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("반복 횟수는 1 이상의 정수여야 합니다.")


def benchmark_sizes(sizes: Sequence[int] = (3, 5, 13, 25), repeats: int = 10) -> List[BenchmarkResult]:
    """Measure generated Cross MAC calls for each requested size."""
    _validate_repeats(repeats)
    results = []
    for size in sizes:
        pattern = generate_cross(size)
        average_ms = _average_mac_time_2d(pattern, pattern, repeats)
        results.append(BenchmarkResult(size, average_ms, repeats, size * size))
    return results


def compare_2d_1d(size: int, repeats: int = 10) -> ComparisonResult:
    """Compare 2D and flattened MAC using identical generated data."""
    _validate_repeats(repeats)
    pattern = generate_cross(size)
    filter_matrix = generate_x(size)
    pattern_flat = [value for row in pattern for value in row]
    filter_flat = [value for row in filter_matrix for value in row]
    return ComparisonResult(
        size=size,
        repeats=repeats,
        score_2d=mac_2d(pattern, filter_matrix),
        score_1d=mac_1d(pattern_flat, filter_flat),
        average_2d_ms=_average_mac_time_2d(pattern, filter_matrix, repeats),
        average_1d_ms=_average_mac_time_1d(pattern_flat, filter_flat, repeats),
    )


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
            try:
                finite = math.isfinite(value)
            except OverflowError as error:
                raise ValueError("행렬 값이 너무 커서 계산할 수 없습니다.") from error
            if not finite:
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
    pattern_width = len(pattern[0])
    if any(
        len(row) != pattern_width or len(filter_matrix[index]) != pattern_width
        for index, row in enumerate(pattern)
    ):
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
