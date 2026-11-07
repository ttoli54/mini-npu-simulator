import unittest
import tempfile
import subprocess
import sys
import os
from pathlib import Path

from npu import (
    EPSILON,
    classify_scores,
    generate_cross,
    generate_x,
    mac_1d,
    mac_2d,
    normalize_label,
    validate_matrix,
)
from npu import analyze_data, benchmark_sizes, compare_2d_1d, load_data


class MacTests(unittest.TestCase):
    def test_mac_2d_multiplies_matching_positions_and_accumulates(self):
        pattern = [[1.0, 2.0], [3.0, 4.0]]
        filter_matrix = [[2.0, 0.5], [1.0, -1.0]]

        self.assertEqual(mac_2d(pattern, filter_matrix), 2.0)

    def test_mac_1d_matches_the_two_dimensional_result(self):
        pattern = [1.0, 2.0, 3.0, 4.0]
        filter_values = [2.0, 0.5, 1.0, -1.0]

        self.assertEqual(mac_1d(pattern, filter_values), 2.0)

    def test_mac_rejects_different_matrix_sizes(self):
        with self.assertRaises(ValueError):
            mac_2d([[1.0]], [[1.0, 2.0]])

    def test_higher_score_wins_and_tiny_difference_is_undecided(self):
        self.assertEqual(classify_scores(5.0, 1.0), "A")
        self.assertEqual(classify_scores(1.0, 5.0), "B")
        self.assertLess(abs(0.9000000000000000 - 0.8999999999999999), EPSILON)
        self.assertEqual(
            classify_scores(0.9000000000000000, 0.8999999999999999),
            "UNDECIDED",
        )


class ValidationAndPatternTests(unittest.TestCase):
    def test_normalize_label_supports_cross_aliases(self):
        self.assertEqual(normalize_label("+"), "Cross")
        self.assertEqual(normalize_label("cross"), "Cross")
        self.assertEqual(normalize_label("x"), "X")

    def test_normalize_label_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            normalize_label("circle")

    def test_validate_matrix_rejects_non_square_and_non_finite_values(self):
        self.assertEqual(validate_matrix([[1.0, 2.0], [3.0, 4.0]], 2), 2)
        with self.assertRaises(ValueError):
            validate_matrix([[1.0, 2.0], [3.0]], 2)
        with self.assertRaises(ValueError):
            validate_matrix([[float("nan")]], 1)
        with self.assertRaises(ValueError):
            validate_matrix([[float("inf")]], 1)

    def test_generators_create_expected_three_by_three_shapes(self):
        self.assertEqual(
            generate_cross(3),
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        )
        self.assertEqual(
            generate_x(3),
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        )

    def test_generators_reject_even_or_too_small_sizes(self):
        for generator in (generate_cross, generate_x):
            with self.assertRaises(ValueError):
                generator(2)
            with self.assertRaises(ValueError):
                generator(0)


class JsonAnalysisTests(unittest.TestCase):
    def test_original_data_has_six_passing_cases_after_label_normalization(self):
        data = load_data(Path(__file__).resolve().parents[1] / "data.json")

        report = analyze_data(data)

        self.assertEqual(report.total, 6)
        self.assertEqual(report.passed, 3)
        self.assertEqual(report.failed, 3)
        self.assertEqual(
            [failure.case_id for failure in report.failures],
            ["size_5_1", "size_13_2", "size_25_1"],
        )
        self.assertEqual(report.cases[0].size_key, "size_5")
        self.assertEqual(report.cases[0].case_id, "size_5_1")

    def test_case_errors_are_recorded_without_stopping_other_cases(self):
        data = {
            "filters": {
                "size_3": {
                    "cross": [[1.0, 0.0, 0.0]] * 3,
                    "x": [[0.0, 1.0, 0.0]] * 3,
                }
            },
            "patterns": {
                "size_3_1": {
                    "input": [[1.0, 0.0, 0.0]] * 3,
                    "expected": "+",
                },
                "bad-key": {
                    "input": [[1.0, 0.0, 0.0]] * 3,
                    "expected": "x",
                },
                "size_3_3": {
                    "input": [[1.0, 0.0], [0.0, 1.0]],
                    "expected": "+",
                },
                "size_5_1": {
                    "input": [[1.0]],
                    "expected": "circle",
                },
            },
        }

        report = analyze_data(data)

        self.assertEqual(report.total, 4)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.failed, 3)
        self.assertEqual([failure.case_id for failure in report.failures], ["bad-key", "size_3_3", "size_5_1"])
        self.assertTrue(all(failure.reason for failure in report.failures))

    def test_load_data_rejects_invalid_json_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "bad.json"
            path.write_text("not json", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_data(path)


class BenchmarkAndCliTests(unittest.TestCase):
    def test_benchmark_reports_ten_repetitions_and_n_squared_operations(self):
        results = benchmark_sizes((3, 5), repeats=10)

        self.assertEqual([result.size for result in results], [3, 5])
        self.assertTrue(all(result.repeats == 10 for result in results))
        self.assertEqual([result.operation_count for result in results], [9, 25])
        self.assertTrue(all(result.average_ms >= 0.0 for result in results))

    def test_one_dimensional_bonus_uses_the_same_score_as_two_dimensional_mac(self):
        comparison = compare_2d_1d(3, repeats=10)

        self.assertEqual(comparison.repeats, 10)
        self.assertEqual(comparison.score_2d, comparison.score_1d)
        self.assertGreaterEqual(comparison.average_2d_ms, 0.0)
        self.assertGreaterEqual(comparison.average_1d_ms, 0.0)

    def test_json_cli_prints_case_routing_benchmark_and_summary(self):
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "main.py"],
            input="2\n",
            text=True,
            encoding="utf-8",
            capture_output=True,
            env=environment,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("size_5_1", result.stdout)
        self.assertIn("size_5_1 → N=5", result.stdout)
        self.assertIn("총 테스트: 6개", result.stdout)
        self.assertIn("통과: 3개", result.stdout)
        self.assertIn("실패: 3개", result.stdout)
        self.assertIn("연산 횟수(N²)", result.stdout)


class UserModeTests(unittest.TestCase):
    def test_read_matrix_retries_bad_length_non_numeric_and_non_finite_rows(self):
        answers = iter(["0 1", "0 a 1", "0 nan 1", "0 1 0", "1 1 1", "0 1 0"])
        messages = []

        from main import read_matrix

        matrix = read_matrix(3, lambda _: next(answers), messages.append)

        self.assertEqual(matrix, [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertEqual(len(messages), 3)

    def test_user_mode_prints_one_result_block_with_scores_timing_and_verdict(self):
        input_data = "\n".join(
            [
                "1",
                "0 1 0",
                "1 1 1",
                "0 1 0",
                "1 0 1",
                "0 1 0",
                "1 0 1",
                "1 0 1",
                "0 1 0",
                "1 0 1",
            ]
        ) + "\n"
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "main.py"],
            input=input_data,
            text=True,
            encoding="utf-8",
            capture_output=True,
            env=environment,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("필터 2개 입력 완료", result.stdout)
        self.assertIn("A 점수: 1.000000", result.stdout)
        self.assertIn("B 점수: 5.000000", result.stdout)
        self.assertIn("평균 처리 시간", result.stdout)
        self.assertIn("최종 판정: B", result.stdout)
        self.assertIn("보너스 1", result.stdout)

    def test_invalid_menu_choice_returns_nonzero(self):
        result = subprocess.run(
            [sys.executable, "main.py"],
            input="9\n",
            text=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
