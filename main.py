"""Console entry point for the Mini NPU simulator."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List

from npu import AnalysisReport, analyze_data, benchmark_sizes, classify_scores, compare_2d_1d, load_data, mac_2d


def _print_benchmark() -> None:
    print("\n[성능 분석: 평균/10회]")
    print("크기(N×N)   평균 시간(ms)   연산 횟수(N²)")
    print("------------------------------------------")
    for result in benchmark_sizes():
        print(f"{result.size}×{result.size:<7} {result.average_ms:>10.4f}       {result.operation_count}")


def _print_bonus_comparison() -> None:
    result = compare_2d_1d(3)
    print("\n[보너스 1: 2차원/1차원 MAC 비교]")
    print(f"같은 점수: {result.score_2d:.6f} / {result.score_1d:.6f}")
    print(f"2차원 평균: {result.average_2d_ms:.4f} ms, 1차원 평균: {result.average_1d_ms:.4f} ms (각 {result.repeats}회)")


def read_matrix(size: int, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> List[List[float]]:
    """Read exactly `size` rows of finite numbers from the console."""
    matrix = []
    for row_number in range(1, size + 1):
        while True:
            raw = input_fn(f"{row_number}행: ")
            parts = raw.strip().split()
            if len(parts) != size:
                output_fn(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
                continue
            try:
                row = [float(value) for value in parts]
            except ValueError:
                output_fn("입력 형식 오류: 숫자만 입력하세요.")
                continue
            if not all(math.isfinite(value) for value in row):
                output_fn("입력 형식 오류: NaN과 무한대는 입력할 수 없습니다.")
                continue
            matrix.append(row)
            break
    return matrix


def run_user_mode(input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> int:
    output_fn("\n[필터 입력]")
    output_fn("필터 A (3줄 입력, 공백 구분)")
    filter_a = read_matrix(3, input_fn, output_fn)
    output_fn("필터 B (3줄 입력, 공백 구분)")
    filter_b = read_matrix(3, input_fn, output_fn)
    output_fn("필터 2개 입력 완료")
    output_fn("\n[패턴 입력]")
    pattern = read_matrix(3, input_fn, output_fn)

    score_a = mac_2d(pattern, filter_a)
    score_b = mac_2d(pattern, filter_b)
    timing = benchmark_sizes((3,), repeats=10)[0]
    verdict = classify_scores(score_a, score_b)
    output_fn("\n[MAC 결과]")
    output_fn(f"A 점수: {score_a:.6f}")
    output_fn(f"B 점수: {score_b:.6f}")
    output_fn(f"평균 처리 시간(10회): {timing.average_ms:.4f} ms")
    output_fn(f"최종 판정: {verdict}")
    _print_bonus_comparison()
    return 0


def _print_report(report: AnalysisReport) -> None:
    print("\n[패턴 분석]")
    for case in report.cases:
        if case.scores:
            size = case.size_key[5:] if case.size_key.startswith("size_") else case.size_key
            print(f"- {case.case_id} → N={size} → {case.size_key}의 Cross/X 필터 사용")
            print(f"  Cross 점수: {case.scores['Cross']:.6f}")
            print(f"  X 점수: {case.scores['X']:.6f}")
            print(f"  판정: {case.predicted} | expected: {case.expected} | {case.status} ({case.reason})")
        else:
            print(f"- {case.case_id}: FAIL ({case.reason})")
    _print_benchmark()
    _print_bonus_comparison()
    print("\n[결과 요약]")
    print(f"총 테스트: {report.total}개")
    print(f"통과: {report.passed}개")
    print(f"실패: {report.failed}개")
    if report.failures:
        print("실패 케이스:")
        for case in report.failures:
            print(f"- {case.case_id}: {case.reason}")


def run_json_mode(data_path: Path) -> int:
    try:
        report = analyze_data(load_data(data_path))
    except (OSError, ValueError) as error:
        print(f"data.json을 읽을 수 없습니다: {error}")
        return 1
    _print_report(report)
    return 0


def main() -> int:
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]\n1. 사용자 입력 (3x3)\n2. data.json 분석")
    choice = input("선택: ").strip()
    if choice == "2":
        data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data.json")
        return run_json_mode(data_path)
    if choice == "1":
        return run_user_mode()
    print("1 또는 2를 선택해주세요.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
