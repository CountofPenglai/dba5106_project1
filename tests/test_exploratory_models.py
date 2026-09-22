import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from portfolio.models.elastic_net_portfolio import fit_weights as fit_elastic_net
from portfolio.models.market_beta_portfolio import calculate_market_betas, fit_portfolio
from portfolio.models.pca_regularized_portfolio import fit_weights as fit_pca


ROOT = Path(__file__).resolve().parents[1]


def history(rows=40, assets=6):
    rng = np.random.default_rng(173)
    common = rng.normal(0.0004, 0.008, (rows, 1))
    noise = rng.normal(0, 0.004, (rows, assets))
    return pd.DataFrame(common * np.linspace(0.6, 1.4, assets) + noise,
                        index=pd.bdate_range("2025-01-02", periods=rows),
                        columns=[f"P{i}" for i in range(assets)])


def write_french_csv(path, frame):
    raw = frame.copy() * 100
    raw.index = raw.index.strftime("%Y%m%d")
    path.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")


def reference_elastic(history, ratio, strength=1e-5):
    r = history.to_numpy()
    assets = r.shape[1]
    ew = np.full(assets, 1 / assets)
    null = np.vstack([np.eye(assets - 1), -np.ones(assets - 1)])
    model = ElasticNet(alpha=strength / 2, l1_ratio=ratio, fit_intercept=True,
                       max_iter=100000, tol=1e-8).fit(r @ null, r @ ew)
    return ew - null @ model.coef_


def reference_pca(history, components, kind, lasso=1e-5, ridge=1e-4):
    r = history.to_numpy()
    assets = r.shape[1]
    ew = np.full(assets, 1 / assets)
    null = np.vstack([np.eye(assets - 1), -np.ones(assets - 1)])
    pca = PCA(n_components=components)
    transformed = pca.fit_transform(r @ null)
    if kind == "LASSO":
        model = Lasso(alpha=lasso / 2, fit_intercept=True,
                      max_iter=100000, tol=1e-8)
    else:
        model = Ridge(alpha=ridge / 2, fit_intercept=True)
    model.fit(transformed, r @ ew)
    return ew - null @ (pca.components_.T @ model.coef_)


class ExploratoryModelTests(unittest.TestCase):
    def test_elastic_net_preserves_legacy_sklearn_strength_and_budget(self):
        sample = history()
        actual = fit_elastic_net(sample, 0.85)
        assert_allclose(actual, reference_elastic(sample, 0.85), rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=13)

    def test_pca_regularized_models_preserve_legacy_projection(self):
        sample = history()
        for kind in ("LASSO", "Ridge"):
            with self.subTest(kind=kind):
                actual = fit_pca(sample, 3, kind)
                assert_allclose(actual, reference_pca(sample, 3, kind), rtol=1e-12, atol=1e-14)
                self.assertAlmostEqual(float(actual.sum()), 1.0, places=13)

    def test_pca_rejects_unknown_regularizer(self):
        with self.assertRaisesRegex(ValueError, "LASSO.*Ridge"):
            fit_pca(history(), 3, "ElasticNet")

    def test_market_beta_matches_covariance_definition(self):
        sample = history()
        betas, benchmark = calculate_market_betas(sample)
        centered_benchmark = benchmark - benchmark.mean()
        expected = ((sample.to_numpy() - sample.to_numpy().mean(axis=0)).T
                    @ centered_benchmark) / np.sum(centered_benchmark ** 2)
        assert_allclose(betas, expected, rtol=1e-13, atol=1e-14)

    def test_market_beta_fit_preserves_constraints_and_status_policy(self):
        sample = history(rows=30, assets=4)
        weights, beta = fit_portfolio(sample, "Ridge", (0.8, 1.2))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertLessEqual(float(np.abs(weights).sum()), 1.5 + 1e-7)
        self.assertGreaterEqual(beta, 0.8 - 1e-6)
        self.assertLessEqual(beta, 1.2 + 1e-6)
        realized = sample.to_numpy() @ weights
        benchmark = sample.mean(axis=1).to_numpy()
        self.assertLessEqual(float(np.var(realized)), float(np.var(benchmark)) + 1e-9)
        self.assertGreaterEqual(float((1 + realized).min()), 1e-6 - 1e-8)

    def test_market_beta_falls_back_only_for_solver_errors(self):
        sample = history(rows=20, assets=4)
        original = cp.Problem.solve
        calls = []

        def fail_clarabel(problem, *args, **kwargs):
            calls.append(kwargs.get("solver"))
            if kwargs.get("solver") == cp.CLARABEL:
                raise cp.error.SolverError("forced")
            return original(problem, *args, **kwargs)

        cp.Problem.solve = fail_clarabel
        try:
            weights, _ = fit_portfolio(sample, "LASSO")
        finally:
            cp.Problem.solve = original
        self.assertEqual(calls[:2], [cp.CLARABEL, cp.SCS])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)

    def test_new_experiment_modules_expose_complete_help(self):
        for module in ("experiments.elastic_net_portfolio",
                       "experiments.pca_regularized_portfolio",
                       "experiments.market_beta_constraints"):
            with self.subTest(module=module):
                run = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                                     capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                for option in ("--csv", "--start", "--end", "--window",
                               "--output", "--layout"):
                    self.assertIn(option, run.stdout)

    def test_legacy_import_alias_keeps_constants_patchable(self):
        import elastic_net_test as legacy
        import experiments.elastic_net_portfolio as recipe
        self.assertIs(legacy, recipe)
        sample = history()
        original = legacy.STRENGTH
        try:
            legacy.STRENGTH = 2e-4
            assert_allclose(legacy.fit_weights(sample, 0.8),
                            reference_elastic(sample, 0.8, strength=2e-4),
                            rtol=1e-12, atol=1e-14)
        finally:
            legacy.STRENGTH = original

    def test_all_recipe_loaders_drop_rows_missing_a_retained_asset(self):
        from experiments.elastic_net_portfolio import load_returns as load_elastic
        from experiments.market_beta_constraints import load_returns as load_market
        from experiments.pca_regularized_portfolio import load_returns as load_pca
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "returns.csv"
            sample = history(rows=5, assets=98)
            missing_date = sample.index[2]
            sample.loc[missing_date, "P47"] = np.nan
            write_french_csv(csv, sample)
            for loader in (load_elastic, load_pca, load_market):
                with self.subTest(loader=loader.__module__):
                    loaded = loader(csv)
                    self.assertNotIn(missing_date, loaded.index)
                    self.assertEqual(len(loaded), 4)

    def test_elastic_net_cli_writes_categorized_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dates = pd.bdate_range("2025-12-01", periods=12)
            raw = history(rows=12, assets=4)
            raw.index = dates
            csv = directory / "returns.csv"
            write_french_csv(csv, raw)
            output = directory / "output"
            run = subprocess.run([
                sys.executable, "-m", "experiments.elastic_net_portfolio",
                "--csv", str(csv), "--start", str(dates[-2].date()),
                "--end", str(dates[-1].date()), "--window", "6",
                "--output", str(output), "--layout", "categorized"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            manifest = json.loads((output / "artifact_manifest.json").read_text())
            self.assertEqual(manifest["artifacts"]["summary.csv"], "metrics/summary.csv")
            self.assertTrue((output / "returns" / "ElasticNet_l1ratio_0.25_returns.csv").is_file())
            self.assertTrue((output / "weights" / "ElasticNet_l1ratio_1_weights.csv").is_file())
            metadata = json.loads((output / "config" / "run_metadata.json").read_text())
            self.assertEqual(metadata["artifact_layout"], "categorized")
            self.assertEqual(len(metadata["data_sha256"]), 64)
            self.assertIn(str(output.resolve()), run.stdout)

    def test_pca_and_market_beta_clis_write_categorized_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sample = history(rows=32, assets=98)
            csv = directory / "returns.csv"
            write_french_csv(csv, sample)
            start, end = sample.index[-2].date(), sample.index[-1].date()
            cases = (
                ("experiments.pca_regularized_portfolio", "pca", "summary.csv"),
                ("experiments.market_beta_constraints", "market", "market_beta_results.csv"),
            )
            for module, folder, logical_summary in cases:
                with self.subTest(module=module):
                    output = directory / folder
                    run = subprocess.run([
                        sys.executable, "-m", module, "--csv", str(csv),
                        "--start", str(start), "--end", str(end), "--window", "30",
                        "--output", str(output), "--layout", "categorized"],
                        cwd=ROOT, capture_output=True, text=True, timeout=120,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                             "OPENBLAS_NUM_THREADS": "1"})
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    manifest = json.loads((output / "artifact_manifest.json").read_text())
                    self.assertEqual(manifest["artifacts"][logical_summary],
                                     f"metrics/{logical_summary}")
                    self.assertTrue(any((output / "returns").iterdir()))
                    self.assertTrue(any((output / "weights").iterdir()))
                    metadata = json.loads((output / "config" / "run_metadata.json").read_text())
                    self.assertEqual(metadata["artifact_layout_version"], 1)
                    self.assertIn("drop every row", metadata["missing_row_policy"])
                    self.assertIn(str(output.resolve()), run.stdout)


if __name__ == "__main__":
    unittest.main()
