"""Capture reproducible command output as browser-rendered evidence pages."""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "docs" / "evidence"


def run_command(args: list[str], input_text: str = "") -> str:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        args,
        cwd=ROOT,
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=environment,
        check=False,
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    output = "\n".join(line.rstrip() for line in output.splitlines())
    output += f"\n[exit code: {result.returncode}]"
    return output


def write_terminal_page(filename: str, title: str, command: str, output: str) -> None:
    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #111827; color: #e5e7eb; font-family: Consolas, 'Malgun Gothic', monospace; }}
main {{ margin: 32px; background: #0b1220; border: 2px solid #334155; border-radius: 14px; padding: 28px; }}
h1 {{ color: #cbd5e1; font-size: 24px; margin: 0 0 22px; }}
.command {{ color: #93c5fd; margin-bottom: 18px; }}
pre {{ white-space: pre; font-size: 17px; line-height: 1.55; margin: 0; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div class="command">PS&gt; {html.escape(command)}</div>
<pre>{html.escape(output)}</pre>
</main></body></html>"""
    (EVIDENCE_DIR / filename).write_text(page, encoding="utf-8")
    (EVIDENCE_DIR / filename.replace(".html", ".txt")).write_text(
        f"실행 명령: {command}\n\n{output}", encoding="utf-8"
    )


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    user_input = "\n".join(
        [
            "1",
            "0 1 0", "1 1 1", "0 1 0",
            "1 0 1", "0 1 0", "1 0 1",
            "1 0 1", "0 1 0", "1 0 1",
        ]
    ) + "\n"
    write_terminal_page(
        "user-mode-run.html",
        "Mini NPU Simulator - 3x3 user mode",
        "(input is provided by the evidence script) | py -3 main.py",
        run_command([sys.executable, "main.py"], user_input),
    )
    write_terminal_page(
        "json-analysis-run.html",
        "Mini NPU Simulator - JSON analysis",
        "(input: 2) | py -3 main.py",
        run_command([sys.executable, "main.py"], "2\n"),
    )
    write_terminal_page(
        "profile-run.html",
        "Mini NPU Simulator - profiling",
        "py -3 profile.py --sizes 3,5,13,25,101,251 --large-size 1001",
        run_command([
            sys.executable,
            "profile.py",
            "--sizes", "3,5,13,25,101,251",
            "--repeats", "10",
            "--large-size", "1001",
            "--large-repeats", "1",
            "--output", "docs/benchmark-report.md",
        ]),
    )
    write_terminal_page(
        "unittest-run.html",
        "Mini NPU Simulator - unit tests",
        "py -3 -m unittest -v",
        run_command([sys.executable, "-m", "unittest", "-v"]),
    )
    print("HTML evidence pages written to docs/evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
