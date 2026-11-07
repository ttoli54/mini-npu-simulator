import unittest

from npu import EPSILON, classify_scores, mac_1d, mac_2d


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


if __name__ == "__main__":
    unittest.main()
