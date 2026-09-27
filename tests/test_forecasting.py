import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from portfolio.config import TrainingConfig
from portfolio.data.history_windows import TrainingWindow


class ForecastingTests(unittest.TestCase):
    def _returns(self, n=90, p=3, start="2021-01-01"):
        dates = pd.bdate_range(start, periods=n)
        values = np.arange(n * p, dtype=float).reshape(n, p) / 10000.0
        return pd.DataFrame(values, index=dates, columns=[f"a{i}" for i in range(p)])

    def test_build_supervised_has_exact_causal_features(self):
        from portfolio.models.lasso_return_forecast import build_supervised

        returns = self._returns(65, 2)
        data = build_supervised(returns)

        self.assertEqual(data.features.shape, (2, 2, 6))
        self.assertTrue(data.targets.equals(returns.iloc[63:]))
        history = returns.iloc[:63]
        expected = np.array([
            history.iloc[-1, 0],
            np.prod(1 + history.iloc[-5:, 0]) - 1,
            np.prod(1 + history.iloc[-21:, 0]) - 1,
            np.prod(1 + history.iloc[-63:, 0]) - 1,
            history.iloc[-21:, 0].std(ddof=1),
            np.prod(1 + history.iloc[-21:].mean(axis=1)) - 1,
        ])
        np.testing.assert_allclose(data.features[0, 0], expected, rtol=0, atol=1e-15)

    def test_build_supervised_rejects_malformed_or_incomplete_grid(self):
        from portfolio.models.lasso_return_forecast import build_supervised

        base = self._returns(70, 2)
        cases = [
            base.iloc[::-1],
            pd.concat([base, base.iloc[[-1]]]),
            base.rename_axis(index=None).set_axis(range(len(base))),
            base.assign(a0=np.nan),
            base.assign(a0=np.inf),
        ]
        for bad in cases:
            with self.subTest(index=type(bad.index), rows=len(bad)):
                with self.assertRaises(ValueError):
                    build_supervised(bad)

    def test_build_supervised_does_not_drop_missing_rows_after_warmup(self):
        from portfolio.models.lasso_return_forecast import build_supervised

        returns = self._returns(80, 2)
        returns.iloc[70, 0] = np.nan
        with self.assertRaises(ValueError):
            build_supervised(returns)

    def test_forecast_matches_independent_sklearn_and_handles_constant_features(self):
        from portfolio.models.lasso_return_forecast import build_supervised, forecast_at

        returns = self._returns(340, 3)
        returns.iloc[:, 2] = 0.0
        data = build_supervised(returns)
        date = data.targets.index[-1]
        config = TrainingConfig(TrainingWindow(observations=252))
        alpha = 1e-5
        got, diagnostics = forecast_at(data, date, config, alpha)

        position = data.targets.index.get_loc(date)
        X = data.features[position - 252:position]
        y = data.targets.iloc[position - 252:position]
        expected = []
        for asset in range(3):
            scaler = StandardScaler().fit(X[:, asset, :])
            model = Lasso(alpha=alpha, max_iter=100000, tol=1e-8).fit(
                scaler.transform(X[:, asset, :]), y.iloc[:, asset]
            )
            expected.append(model.predict(scaler.transform(data.features[position, asset, :][None]))[0])
        np.testing.assert_allclose(got.to_numpy(), expected, rtol=0, atol=1e-14)
        self.assertEqual(diagnostics["train_count"], 252)
        self.assertEqual(diagnostics["train_end"], data.targets.index[-2])
        self.assertEqual(diagnostics["constant_feature_counts"]["a2"], 5)

    def test_calendar_window_includes_anniversary_and_excludes_forecast_date(self):
        from portfolio.models.lasso_return_forecast import build_supervised, forecast_at

        data = build_supervised(self._returns(900, 2, "2021-01-04"))
        date = data.targets.index[-1]
        config = TrainingConfig(TrainingWindow(years=3, min_observations=252))
        _, diagnostics = forecast_at(data, date, config, 1e-5)
        expected_start = data.targets.index[data.targets.index.searchsorted(date - pd.DateOffset(years=3))]
        self.assertEqual(diagnostics["train_start"], expected_start)
        self.assertLess(diagnostics["train_end"], date)

    def test_future_perturbation_does_not_change_earlier_forecast(self):
        from portfolio.models.lasso_return_forecast import build_supervised, forecast_at

        returns = self._returns(400, 2)
        date = returns.index[350]
        config = TrainingConfig(TrainingWindow(observations=252))
        first = forecast_at(build_supervised(returns), date, config, 1e-5)[0]
        changed = returns.copy()
        changed.loc[changed.index >= date] += 100.0
        second = forecast_at(build_supervised(changed), date, config, 1e-5)[0]
        pd.testing.assert_series_equal(first, second)

    def test_calibration_scores_each_fold_and_alpha_and_uses_larger_tie(self):
        from portfolio.models.lasso_return_forecast import build_supervised, calibrate_alpha

        returns = pd.DataFrame(
            0.0,
            index=pd.bdate_range("2021-01-01", periods=180),
            columns=["a", "b"],
        )
        data = build_supervised(returns)
        alphas = [1e-4, 1e-3]
        selected, scores = calibrate_alpha(
            data,
            TrainingConfig(TrainingWindow(observations=30)),
            end=data.targets.index[-1],
            alphas=alphas,
            n_splits=3,
            test_size=10,
        )
        self.assertEqual(selected, 1e-3)
        self.assertEqual(len(scores), 6)
        self.assertEqual(set(scores.columns), {
            "alpha", "fold", "mse", "train_start", "train_end", "train_count",
            "validation_start", "validation_end", "validation_count",
        })
        self.assertTrue((scores["train_end"] < scores["validation_start"]).all())
        self.assertTrue((scores["validation_count"] == 10).all())

    def test_calibration_rejects_bad_end_alpha_and_fold_grid(self):
        from portfolio.models.lasso_return_forecast import build_supervised, calibrate_alpha

        data = build_supervised(self._returns(120, 2))
        config = TrainingConfig(TrainingWindow(observations=30))
        for kwargs in [
            {"end": "1999-01-01"},
            {"end": data.targets.index[-1], "alphas": [0.0]},
            {"end": data.targets.index[-1], "n_splits": 0},
            {"end": data.targets.index[40], "n_splits": 3, "test_size": 20},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    calibrate_alpha(data, config, **kwargs)


if __name__ == "__main__":
    unittest.main()
