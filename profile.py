"""Run reproducible performance and memory measurements for the simulator."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from npu import ProfileResult, profile_size


def _parse_sizes(raw: str) -> list[int]:
    sizes = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("size를 하나 이상 입력하세요.")
    return sizes


def _format_result(result: ProfileResult) -> str:
    return (
        f"{result.size:>4} | {result.repeats:>7} | {result.operation_count:>10} | "
        f"{result.average_2d_ms:>10.4f} | {result.average_1d_ms:>10.4f} | "
        f"{result.peak_2d_kib:>10.2f} | {result.peak_1d_kib:>10.2f} | "
        f"{result.score_2d:.6f}"
    )


def build_report(results: list[ProfileResult]) -> str:
    lines = [
        "# Mini NPU 성능·메모리 프로파일링 결과",
        "",
        f"- Python: {platform.python_version()}",
        f"- OS: {platform.platform()}",
        "- 시간: `time.perf_counter()`로 MAC 반복 구간만 측정",
        "- 메모리: `tracemalloc`으로 표현 생성 시 peak 측정",
        "",
        "| N | 반복 | 연산 수(N²) | 2D 평균(ms) | 1D 평균(ms) | 2D peak(KiB) | 1D peak(KiB) | 점수 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {_format_result(result)} |" for result in results)
    lines.extend(
        [
            "",
            "모든 행에서 2D와 1D MAC 점수가 일치해야 하며, 불일치하면 프로파일링을 실패시킵니다.",
            "작은 입력은 10회, 대규모 입력은 실행 시간을 줄이기 위해 1회로 측정할 수 있습니다. 결과는 실행 환경에 따라 달라지므로 동일 환경에서 전후 비교에 사용합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_parse_sizes, default=[3, 5, 13, 25, 101, 251])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--large-size", type=int, default=1001)
    parser.add_argument("--large-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Markdown 결과 파일 경로")
    args = parser.parse_args()

    results = [profile_size(size, args.repeats) for size in args.sizes]
    if args.large_size not in args.sizes:
        results.append(profile_size(args.large_size, args.large_repeats))
    report = build_report(results)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
