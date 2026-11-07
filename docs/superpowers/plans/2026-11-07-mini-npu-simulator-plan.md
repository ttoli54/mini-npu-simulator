# Mini NPU Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 원본 `data.json`을 분석하고 3×3 직접 입력, MAC 판정, 성능 측정, 1차원 최적화 비교, Cross/X 생성까지 제공하는 표준 라이브러리 기반 콘솔 앱을 만든다.

**Architecture:** `main.py`는 메뉴와 출력만 담당하고 `npu.py`는 순수 계산·검증·JSON 분석·벤치마크를 담당한다. 테스트는 `tests/test_npu.py`에서 실제 함수와 임시 JSON을 사용해 검증하며, CLI 테스트는 `subprocess`로 두 모드를 확인한다.

**Tech Stack:** Python 3.8+, 표준 라이브러리(`json`, `math`, `re`, `time`, `pathlib`, `statistics`, `unittest`), 외부 패키지 없음

**Spec:** `docs/superpowers/specs/2026-11-07-mini-npu-simulator-design.md`

## Global Constraints

- MAC은 NumPy 없이 반복문으로 직접 구현한다.
- `EPSILON = 1e-9`, 동점 조건은 `abs(a - b) < EPSILON`이다.
- 모드 1은 3×3 숫자 행을 공백으로 입력받고 잘못된 행은 재입력시킨다.
- 모드 2는 `size_N_번호` 키에서 N을 추출하고 `filters[size_N]`을 선택한다.
- `+`/`cross`는 `Cross`, `x`는 `X`로 정규화한다.
- 케이스 오류는 FAIL로 기록하고 다음 케이스를 계속한다.
- 벤치마크는 3·5·13·25를 각각 10회 측정하고 평균 ms와 N²를 출력한다.
- 보너스 1은 같은 입력·반복 횟수의 `mac_2d`/`mac_1d` 비교다.
- 보너스 2는 홀수 N≥3 Cross/X 생성기다.

## Review Focus

- NaN·무한대 입력: 모드 1과 JSON 숫자를 거부한다. (Task 2)
- 잘못된 패턴 키·크기·필터 라벨: 전체 실행을 중단하지 않고 해당 케이스 FAIL로 기록한다. (Task 3)
- 점수 차이가 EPSILON 미만인 부동소수점 동점: `UNDECIDED`가 된다. (Task 1)
- 원본 작업 폴더가 아닌 곳에서 실행: `data.json` 기본 경로가 `main.py` 기준으로 해석된다. (Task 4)
- 짝수·0·음수 패턴 생성 요청: 명확한 `ValueError`로 거부한다. (Task 5)

### Task 1: MAC primitives and verdict rules

**Files:**
- Create: `npu.py`
- Create: `tests/test_npu.py`

**Interfaces:**
- Produces `EPSILON: float`, `mac_2d(pattern, filter_matrix) -> float`, `mac_1d(pattern, filter_matrix) -> float`, `classify_scores(score_a, score_b, label_a="A", label_b="B") -> str`.

- [ ] **Step 1: Write failing tests** for equal-size 2D MAC, size mismatch, 1D-equivalent MAC, higher-score verdict, and epsilon tie.
- [ ] **Step 2: Run `python -m unittest tests.test_npu -v` and confirm the new imports/functions fail.**
- [ ] **Step 3: Implement the two nested-loop MAC functions and centralized verdict rule in `npu.py`.** Validate rectangular/square equal dimensions and raise `ValueError` for mismatch.
- [ ] **Step 4: Run the focused tests and confirm PASS.**
- [ ] **Step 5: Commit `test: define MAC and verdict behavior` or equivalent focused commit.**

### Task 2: Matrix, label, and generated-pattern validation

**Files:**
- Modify: `npu.py`
- Test: `tests/test_npu.py`

**Interfaces:**
- Produces `normalize_label(value) -> str`, `validate_matrix(matrix, expected_size=None) -> int`, `generate_cross(size) -> list[list[float]]`, `generate_x(size) -> list[list[float]]`.

- [ ] **Step 1: Write failing tests** for `+`/`cross`→`Cross`, `x`→`X`, unknown labels, NaN/infinity rejection, non-square rows, and generated odd Cross/X shapes.
- [ ] **Step 2: Run the focused tests and confirm expected failures.**
- [ ] **Step 3: Implement finite-number and square-matrix validation plus odd-size Cross/X generation.** The center is included in both shapes.
- [ ] **Step 4: Run the focused tests and confirm PASS.**
- [ ] **Step 5: Commit `feat: add matrix labels and pattern generator`.**

### Task 3: JSON loading, case analysis, and report model

**Files:**
- Modify: `npu.py`
- Test: `tests/test_npu.py`

**Interfaces:**
- Produces `load_data(path: Path) -> dict`, `analyze_data(data: dict) -> AnalysisReport`, and dataclasses or equivalent records for case results and failures.
- `AnalysisReport` exposes `total`, `passed`, `failed`, `cases`, `failures` and preserves case identifiers and reasons.

- [ ] **Step 1: Write failing tests** using the provided data copy and temporary malformed data for six normal cases, size extraction, filter selection logging data, PASS/FAIL counts, malformed key, missing filter, unknown label, and size mismatch.
- [ ] **Step 2: Run focused tests and confirm failures before implementation.**
- [ ] **Step 3: Implement JSON parsing and case-by-case analysis.** Count malformed pattern entries, normalize expected/filter labels, use `classify_scores`, and continue after per-case errors. File read/JSON root errors remain fatal.
- [ ] **Step 4: Run focused tests and confirm PASS.**
- [ ] **Step 5: Commit `feat: analyze JSON pattern cases`.**

### Task 4: Benchmarking and CLI mode 2

**Files:**
- Modify: `npu.py`
- Create: `main.py`
- Test: `tests/test_npu.py`

**Interfaces:**
- Produces `benchmark_sizes(sizes=(3, 5, 13, 25), repeats=10) -> list[BenchmarkResult]` and `compare_2d_1d(size, repeats=10) -> ComparisonResult`.
- `main.py` accepts menu choice `1` or `2`; mode 2 defaults to `Path(__file__).with_name("data.json")` and supports an optional path argument.

- [ ] **Step 1: Write failing tests** for 10 repetitions, N² operation counts, same generated input in 2D/1D comparison, CLI mode 2 summary, and default data path resolution.
- [ ] **Step 2: Run focused tests and confirm failures.**
- [ ] **Step 3: Implement `perf_counter()` benchmark functions and mode 2 output.** Keep file IO and printing outside timed MAC calls; print case routing, scores, verdict, expected, reason, benchmark table, bonus comparison, and summary.
- [ ] **Step 4: Run focused tests and confirm PASS.**
- [ ] **Step 5: Commit `feat: add JSON mode and MAC benchmarks`.**

### Task 5: CLI mode 1 and complete bonus integration

**Files:**
- Modify: `main.py`, `npu.py`
- Test: `tests/test_npu.py`

**Interfaces:**
- Produces `run_user_mode(input_fn=input, output_fn=print) -> int` and a reusable `read_matrix(size, input_fn, output_fn) -> list[list[float]]`.

- [ ] **Step 1: Write failing tests** for retrying bad row length, non-numeric input, NaN/infinity, successful A/B result, single result block, 3×3 timing, and invalid menu choice.
- [ ] **Step 2: Run focused tests and confirm failures.**
- [ ] **Step 3: Implement robust 3×3 input flow and mode 1 output.** Use A/B labels in mode 1, show both scores, average ms, repeat count, and verdict; run the shared generated benchmark and 2D/1D bonus comparison.
- [ ] **Step 4: Run the complete suite and both CLI modes.** Expected: all tests pass, mode 2 reports the six original cases, and mode 1 accepts the documented Cross/X example.
- [ ] **Step 5: Commit `feat: complete Mini NPU CLI and bonuses`.**

### Task 6: README result report and final verification

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-11-07-mini-npu-simulator-design.md` if implementation decisions differ

- [ ] **Step 1: Run `python main.py` in both modes and capture actual output.**
- [ ] **Step 2: Write README with commands, schema/label rules, MAC explanation, epsilon policy, case report, measured benchmark table, 2D/1D comparison, bonus generator examples, failure analysis, and at least 10 lines of complexity/report discussion.** Do not invent timing values; use the actual run or explain that timings vary.
- [ ] **Step 3: Run `python -m unittest -v`, `python main.py` mode 1, `python main.py` mode 2`, and `git diff --check`.**
- [ ] **Step 4: Review the final diff against the design and plan; fix only verified issues.**
- [ ] **Step 5: Commit `docs: document Mini NPU simulator results`.**
