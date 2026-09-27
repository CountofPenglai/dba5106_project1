import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.data import history_windows as splits
from portfolio.engine.rolling_backtest import backtest
from portfolio.regularization import Regularization
from portfolio.engine.portfolio_fit import fit_portfolio


class TrainingWindowTests(unittest.TestCase):
    def window(self, **kwargs):
        self.assertTrue(hasattr(splits, "TrainingWindow"), "Calendar window configuration is missing")
        return splits.TrainingWindow(**kwargs)

    def test_observation_windows_preserve_126_and_252_histories(self):
        dates = pd.bdate_range("2023-01-02", periods=270)
        frame = pd.DataFrame({"A": np.arange(270)}, index=dates)
        for count in (126, 252):
            old = list(splits.rolling_windows(frame, start=dates[260], end=dates[261], window=count))
            new = list(splits.rolling_windows(frame, start=dates[260], end=dates[261],
                                            window=self.window(observations=count)))
            for offset, ((date, history), (_, legacy)) in enumerate(zip(new, old)):
                self.assertEqual(date, dates[260 + offset])
                pd.testing.assert_frame_equal(history, frame.iloc[260 + offset - count:260 + offset])
                pd.testing.assert_frame_equal(history, legacy)
        default = next(splits.rolling_windows(frame, start=dates[260], end=dates[261]))[1]
        self.assertEqual(len(default), 126)

    def test_calendar_cutoff_includes_anniversary_and_handles_leap_day(self):
        dates = pd.to_datetime(["2023-02-27", "2023-02-28", "2023-03-01",
                                "2024-02-28", "2024-02-29", "2024-03-01"])
        frame = pd.DataFrame({"A": range(6)}, index=dates)
        result = list(splits.rolling_windows(frame, start="2024-02-29", end="2024-03-01",
                                            window=self.window(years=1)))
        self.assertEqual(list(result[0][1]["A"]), [1, 2, 3])
        self.assertEqual(list(result[1][1]["A"]), [2, 3, 4])

    def test_calendar_minimum_prevents_silent_short_training(self):
        frame = pd.DataFrame({"A": range(4)}, index=pd.date_range("2026-01-01", periods=4))
        with self.assertRaises(ValueError):
            list(splits.rolling_windows(frame, start="2026-01-03", end="2026-01-04",
                                        window=self.window(years=3, min_observations=3)))
        history = next(splits.rolling_windows(frame, start="2026-01-04", end="2026-01-04",
                                             window=self.window(years=3, min_observations=3)))[1]
        self.assertEqual(list(history["A"]), [0, 1, 2])

    def test_invalid_or_ambiguous_window_settings_are_rejected(self):
        self.window(years=3)
        for options in ({}, {"years": 3, "observations": 252}, {"years": 0},
                        {"years": True}, {"years": 1.5}, {"observations": 1},
                        {"observations": False}, {"years": 3, "min_observations": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                splits.TrainingWindow(**options)

    def test_calendar_backtest_cannot_see_current_or_future_returns(self):
        dates = pd.bdate_range("2022-12-01", "2026-01-06")
        frame = pd.DataFrame(np.random.default_rng(4).normal(0, .01, (len(dates), 3)),
                             index=dates, columns=list("ABC"))
        window = self.window(years=3, min_observations=252)
        strategy = lambda history: fit_portfolio(history, regularization=Regularization("lasso", 1e-5)).weights
        before = backtest(frame, strategy, start="2026-01-02", end="2026-01-06", window=window)
        changed = frame.copy()
        changed.loc["2026-01-05":] += 10
        after = backtest(changed, strategy, start="2026-01-02", end="2026-01-06", window=window)
        assert_allclose(before.weights.loc[:"2026-01-05"], after.weights.loc[:"2026-01-05"])
        assert_allclose(before.returns.loc[:"2026-01-02"], after.returns.loc[:"2026-01-02"])


if __name__ == "__main__":
    unittest.main()
