import unittest

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


if __name__ == "__main__":
    unittest.main()
