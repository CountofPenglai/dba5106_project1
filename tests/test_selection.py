import unittest

import numpy as np

from portfolio.selection import (
    COARSE_PARAMETERS,
    GrowthParameters,
    SearchConfig,
    search_growth,
)


def metrics(cumulative_return, gross=1.1, **extra):
    return {
        "cumulative_return": cumulative_return,
        "average_gross_leverage": gross,
        **extra,
    }


class TabulatedEvaluator:
    def __init__(self, table, default=None):
        self.table = table
        self.default = default
        self.calls = []

    def __call__(self, parameters):
        self.calls.append(parameters)
        value = self.table.get(parameters, self.default)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            return value(parameters)
        if value is None:
            raise AssertionError(f"Unexpected candidate {parameters}")
        return value


class SelectionTests(unittest.TestCase):
    def test_complete_coarse_batch_has_callback_order_ids_and_metrics(self):
        table = {
            parameters: metrics(index / 100, gross=1 + index / 100, label=f"row-{index}")
            for index, parameters in enumerate(COARSE_PARAMETERS)
        }
        evaluate = TabulatedEvaluator(table)

        result = search_growth(evaluate, config=SearchConfig(max_candidates=9))

        self.assertEqual(evaluate.calls, list(COARSE_PARAMETERS))
        self.assertEqual(list(result.history["candidate_id"]), list(range(1, 10)))
        self.assertEqual(list(result.history["round"]), [0] * 9)
        self.assertEqual(list(result.history["status"]), ["ok"] * 9)
        self.assertEqual(list(result.history["label"]), [f"row-{index}" for index in range(9)])
        self.assertEqual(result.raw_best, COARSE_PARAMETERS[-1])
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual(result.rounds, 0)

    def test_plateau_stops_only_after_the_complete_neighbor_batch(self):
        center = GrowthParameters(.003, .1)
        control = GrowthParameters(0, 0)
        expected_neighbors = [
            GrowthParameters(.0015, .1),
            GrowthParameters(.006, .1),
            GrowthParameters(.003, .05),
            GrowthParameters(.003, .2),
        ]
        evaluate = TabulatedEvaluator(
            {center: metrics(.10), control: metrics(.05)},
            default=metrics(.09),
        )

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=6, patience=1),
            initial_parameters=[center, control],
        )

        self.assertEqual(evaluate.calls, [center, control, *expected_neighbors])
        self.assertEqual(list(result.history["round"]), [0, 0, 1, 1, 1, 1])
        self.assertEqual(result.stop_reason, "patience")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.raw_best, center)

    def test_zero_and_upper_bound_neighbors_follow_the_fixed_search_space(self):
        cases = [
            (
                GrowthParameters(0, 0),
                [GrowthParameters(.0015, 0), GrowthParameters(0, .05)],
            ),
            (
                GrowthParameters(.03, 3),
                [GrowthParameters(.015, 3), GrowthParameters(.03, 1.5)],
            ),
        ]
        for center, neighbors in cases:
            with self.subTest(center=center):
                control = GrowthParameters(.01, .1)
                evaluate = TabulatedEvaluator(
                    {center: metrics(.10), control: metrics(.05)},
                    default=metrics(.09),
                )

                result = search_growth(
                    evaluate,
                    config=SearchConfig(max_candidates=4, patience=1),
                    initial_parameters=[center, control],
                )

                self.assertEqual(evaluate.calls, [center, control, *neighbors])
                self.assertEqual(result.stop_reason, "patience")

    def test_budget_never_starts_a_partial_neighbor_batch(self):
        center = GrowthParameters(.003, .1)
        control = GrowthParameters(0, 0)
        evaluate = TabulatedEvaluator({center: metrics(.10), control: metrics(.05)})

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=5),
            initial_parameters=[center, control],
        )

        self.assertEqual(evaluate.calls, [center, control])
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual(result.rounds, 0)

    def test_zero_threshold_still_requires_an_actual_improvement(self):
        center = GrowthParameters(.003, .1)
        control = GrowthParameters(0, 0)
        evaluate = TabulatedEvaluator(
            {center: metrics(.10), control: metrics(.05)},
            default=metrics(.10),
        )

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=10, min_improvement=0, patience=1),
            initial_parameters=[center, control],
        )

        self.assertEqual(result.stop_reason, "patience")
        self.assertEqual(result.rounds, 1)

    def test_small_gains_accumulate_against_the_last_significant_anchor(self):
        start = GrowthParameters(.003, .1)
        control = GrowthParameters(.01, 1)
        table = {
            start: metrics(.100),
            control: metrics(.060),
            GrowthParameters(.006, .1): metrics(.1010),
            GrowthParameters(.012, .1): metrics(.1025),
            GrowthParameters(.024, .1): metrics(.1034),
            GrowthParameters(.024, .2): metrics(.1048),
        }
        evaluate = TabulatedEvaluator(table, default=metrics(.05))

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=15, min_improvement=.002, patience=2),
            initial_parameters=[start, control],
        )

        self.assertIn(GrowthParameters(.024, .2), evaluate.calls)
        self.assertEqual(result.raw_best, GrowthParameters(.024, .2))
        self.assertEqual(result.rounds, 5)
        self.assertEqual(result.stop_reason, "budget")

    def test_exact_decimal_threshold_resets_patience_for_the_next_round(self):
        start = GrowthParameters(.003, .1)
        control = GrowthParameters(.01, 1)
        boundary = GrowthParameters(.006, .1)
        later_winner = GrowthParameters(.012, .1)
        evaluate = TabulatedEvaluator(
            {
                start: metrics(.100),
                control: metrics(.060),
                boundary: metrics(.102),
                later_winner: metrics(.103),
            },
            default=metrics(.05),
        )

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=9, min_improvement=.002, patience=1),
            initial_parameters=[start, control],
        )

        self.assertIn(later_winner, evaluate.calls)
        self.assertEqual(result.raw_best, later_winner)
        self.assertEqual(result.rounds, 2)

    def test_duplicate_initial_parameters_are_evaluated_once(self):
        only = GrowthParameters(0, 0)
        evaluate = TabulatedEvaluator({only: metrics(.1)})

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=1),
            initial_parameters=[only, only],
        )

        self.assertEqual(evaluate.calls, [only])
        self.assertEqual(list(result.history["candidate_id"]), [1])
        self.assertEqual(result.stop_reason, "budget")

    def test_final_selection_uses_conservative_ties_and_restores_recorded_best(self):
        raw = GrowthParameters(.01, 1)
        lower_eta = GrowthParameters(.003, 1)
        lower_gross = GrowthParameters(.003, .1)
        lower_gamma = GrowthParameters(.003, 0)
        table = {
            raw: metrics(.1200, gross=1.05),
            lower_eta: metrics(.1185, gross=1.40),
            lower_gross: metrics(.1184, gross=1.20),
            lower_gamma: metrics(.1181, gross=1.20),
        }
        evaluate = TabulatedEvaluator(table)

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=4, min_improvement=.002),
            initial_parameters=[raw, lower_eta, lower_gross, lower_gamma],
        )

        self.assertEqual(result.raw_best, raw)
        self.assertEqual(result.selected, lower_gamma)
        self.assertAlmostEqual(result.raw_best_score, .1200)
        self.assertAlmostEqual(result.selected_score, .1181)

    def test_numerical_failures_and_invalid_metrics_are_recorded(self):
        good = GrowthParameters(0, 0)
        runtime = GrowthParameters(.003, 0)
        invalid = GrowthParameters(.01, 0)
        missing = GrowthParameters(0, .1)
        evaluate = TabulatedEvaluator({
            good: metrics(.1, sharpe=.7),
            runtime: RuntimeError("solver did not converge"),
            invalid: metrics(np.nan),
            missing: {"candidate_id": 999, "cumulative_return": .2},
        })

        result = search_growth(
            evaluate,
            config=SearchConfig(max_candidates=4),
            initial_parameters=[good, runtime, invalid, missing],
        )

        self.assertEqual(list(result.history["status"]), ["ok", "failed", "failed", "failed"])
        self.assertEqual(list(result.history["candidate_id"]), [1, 2, 3, 4])
        self.assertEqual(result.history.loc[0, "sharpe"], .7)
        self.assertIn("solver did not converge", result.history.loc[1, "error"])
        self.assertIn("finite", result.history.loc[2, "error"])
        self.assertIn("average_gross_leverage", result.history.loc[3, "error"])
        self.assertEqual(result.selected, good)

    def test_all_initial_numerical_failures_raise_after_the_complete_batch(self):
        first = GrowthParameters(0, 0)
        second = GrowthParameters(.003, 0)
        evaluate = TabulatedEvaluator({
            first: RuntimeError("solver failure"),
            second: ValueError("infeasible"),
        })

        with self.assertRaisesRegex(RuntimeError, "All initial candidates failed"):
            search_growth(
                evaluate,
                config=SearchConfig(max_candidates=2),
                initial_parameters=[first, second],
            )

        self.assertEqual(evaluate.calls, [first, second])

    def test_unexpected_callback_exceptions_propagate(self):
        candidate = GrowthParameters(0, 0)
        evaluate = TabulatedEvaluator({candidate: ZeroDivisionError("programmer error")})

        with self.assertRaisesRegex(ZeroDivisionError, "programmer error"):
            search_growth(
                evaluate,
                config=SearchConfig(max_candidates=1),
                initial_parameters=[candidate],
            )

    def test_configuration_parameters_and_initial_batch_are_validated(self):
        for kwargs in (
            {"max_candidates": 0},
            {"max_candidates": True},
            {"min_improvement": -1},
            {"min_improvement": np.inf},
            {"patience": 0},
            {"patience": 1.5},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    SearchConfig(**kwargs)

        for args in ((-1, 0), (np.nan, 0), (.031, 0), (0, 3.01)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    GrowthParameters(*args)

        with self.assertRaisesRegex(ValueError, "initial_parameters"):
            search_growth(lambda _: metrics(0), initial_parameters=[])
        with self.assertRaisesRegex(ValueError, "Initial batch"):
            search_growth(
                lambda _: metrics(0),
                config=SearchConfig(max_candidates=1),
                initial_parameters=[GrowthParameters(0, 0), GrowthParameters(.003, 0)],
            )


if __name__ == "__main__":
    unittest.main()
