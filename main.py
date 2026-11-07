"""Console entry point for the Mini NPU simulator."""

from __future__ import annotations

import sys
from pathlib import Path

from npu import AnalysisReport, analyze_data, benchmark_sizes, compare_2d_1d, load_data


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
        print("사용자 입력 모드는 다음 단계에서 구현합니다.")
        return 0
    print("1 또는 2를 선택해주세요.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
