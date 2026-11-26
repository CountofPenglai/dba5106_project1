import unittest
import numpy as np
import pandas as pd


class ForecastWinnerEvaluationTests(unittest.TestCase):
    def test_scores_use_asset_means_and_predictions_determine_the_picks(self):
        from portfolio.evaluation.forecast_metrics import evaluate_method
        names = ['A', 'B', 'C', 'D']
        actual = pd.DataFrame([[2, 0, 1, -1], [-1, 1, 2, 0], [1, 3, -2, 2]], columns=names) / 100
        pred = pd.DataFrame([[2, 2, 2, 1], [1, 2, 3, 0], [0, 0, 1, 1]], columns=names) / 100
        mean = pd.DataFrame(0., index=actual.index, columns=names)
        stats, daily = evaluate_method(actual, pred, mean, count=2)
        np.testing.assert_allclose(daily.top_return, [.01, .015, 0.])
        np.testing.assert_array_equal(daily.hits, [1, 2, 1])
        squared_error = np.sum(np.square(actual.to_numpy()-pred.to_numpy()))
        self.assertAlmostEqual(stats['pooled_r2'], 1 - squared_error / sum(
            sum((x-x.mean())**2) for _, x in actual.items()))
        self.assertAlmostEqual(stats['skill_vs_trailing_mean'], 1 - squared_error / (actual**2).sum().sum())
        self.assertAlmostEqual(evaluate_method(actual, mean, mean, count=2)[0]['skill_vs_trailing_mean'], 0.)
