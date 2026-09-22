import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.constraints import PortfolioConstraints
from portfolio.data.asset_graph import build_size_bm_incidence
from portfolio.data.returns_loader import load_returns
from portfolio.engine.grid_smoothness_workflow import fit_portfolio_growth_smooth
from portfolio.engine.portfolio_fit import fit_portfolio_growth
from portfolio.engine.rolling_backtest import backtest
from portfolio.losses import LogGrowthLoss
from portfolio.regularization import Regularization

ROOT = Path(__file__).resolve().parents[1]


def synthetic_history(rows=40, assets=8, seed=173):
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0004, 0.008, (rows, 1))
    noise = rng.normal(0, 0.004, (rows, assets))
    columns = [f"P{i}" for i in range(assets)]
    return pd.DataFrame(common * np.linspace(0.6, 1.4, assets) + noise,
                        index=pd.bdate_range("2025-01-02", periods=rows), columns=columns)


def chain_incidence(assets):
    """A simple path graph P0-P1-...-Pn-1, independent of the Size-BM naming."""
    incidence = np.zeros((assets - 1, assets))
    for row in range(assets - 1):
        incidence[row, row], incidence[row, row + 1] = 1.0, -1.0
    return incidence


class SizeBmIncidenceTests(unittest.TestCase):
    def test_real_grid_has_176_edges_and_zero_row_sums(self):
        returns = load_returns(ROOT / "100_Portfolios_10x10_Daily.csv")
        incidence, edges = build_size_bm_incidence(returns.columns)
        self.assertEqual(returns.shape[1], 98)
        self.assertEqual(len(edges), 176)
        self.assertEqual(incidence.shape, (176, 98))
        assert_allclose(incidence.sum(axis=1), 0)
        self.assertTrue(np.all(np.count_nonzero(incidence, axis=1) == 2))

    def test_no_edge_drawn_across_a_missing_grid_position(self):
        columns = ["SMALL LoBM", "ME1 BM2", "ME2 BM1"]  # ME1 BM1, ME1 BM2, ME2 BM1: an L-shape
        _, edges = build_size_bm_incidence(columns)
        # (0,1) horizontal-free but SMALL LoBM=ME1BM1 connects to ME1BM2 (index1) and ME2BM1 (index2).
        self.assertEqual(set(map(tuple, edges)), {(0, 1), (0, 2)})

    def test_unrecognized_name_raises(self):
        with self.assertRaises(ValueError):
            build_size_bm_incidence(["SMALL LoBM", "Not A Real Portfolio"])

    def test_duplicate_names_raise(self):
        with self.assertRaises(ValueError):
            build_size_bm_incidence(["SMALL LoBM", "SMALL LoBM"])

    def test_duplicate_grid_position_raises(self):
        with self.assertRaises(ValueError):
            build_size_bm_incidence(["SMALL LoBM", "ME1 BM1"])


class FitGrowthSmoothTests(unittest.TestCase):
    def test_rho_zero_matches_fit_growth_exactly(self):
        history = synthetic_history()
        constraints = PortfolioConstraints()
        loss = LogGrowthLoss(reward_strength=0, exposure_strength=2)
        regularization = Regularization("lasso", 1e-5)
        baseline = fit_portfolio_growth(history, regularization=regularization, loss=loss, constraints=constraints)
        smoothed = fit_portfolio_growth_smooth(
            history, regularization=regularization, loss=loss, constraints=constraints,
            rho=0.0, incidence=chain_incidence(history.shape[1]))
        assert_allclose(smoothed.weights.to_numpy(), baseline.weights.to_numpy(), atol=1e-6)
        self.assertAlmostEqual(smoothed.training_loss, baseline.training_loss, places=6)

    def test_rho_requires_incidence(self):
        history = synthetic_history()
        with self.assertRaises(ValueError):
            fit_portfolio_growth_smooth(history, rho=0.1, incidence=None)

    def test_negative_rho_raises(self):
        history = synthetic_history()
        with self.assertRaises(ValueError):
            fit_portfolio_growth_smooth(history, rho=-1, incidence=chain_incidence(history.shape[1]))

    def test_positive_rho_shrinks_neighbor_gap_on_a_chain(self):
        history = synthetic_history()
        constraints = PortfolioConstraints()
        loss = LogGrowthLoss(reward_strength=0, exposure_strength=2)
        regularization = Regularization("lasso", 1e-5)
        incidence = chain_incidence(history.shape[1])
        gaps = []
        for rho in (0.0, 0.1, 1.0, 10.0):
            fitted = fit_portfolio_growth_smooth(history, regularization=regularization, loss=loss,
                                                 constraints=constraints, rho=rho, incidence=incidence)
            gaps.append(fitted.diagnostics["neighbor_gap"])
        for earlier, later in zip(gaps, gaps[1:]):
            self.assertLessEqual(later, earlier + 1e-9)
        self.assertLess(gaps[-1], gaps[0])


class GridSmoothnessCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_grid_smoothness.py"), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})

    def test_rho_zero_reproduces_selected_lasso_via_backtest(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            returns = load_returns(ROOT / "100_Portfolios_10x10_Daily.csv")
            dates = returns.index
            start_idx = dates.get_indexer([pd.Timestamp("2026-01-02")], method="bfill")[0]
            train_start = dates[max(0, start_idx - 260)]
            eval_end = dates[start_idx + 30]
            raw = (returns.loc[train_start:eval_end] * 100)
            raw.index = raw.index.strftime("%Y%m%d")
            csv = directory / "input.csv"
            csv.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")
            output = directory / "output"
            run = self.run_cli("--csv", csv, "--start", "2026-01-02", "--end", str(eval_end.date()),
                               "--window", "252", "--rho", "0", "0.1", "--output", output)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            self.assertIn("Selected_LASSO_rho0", summary.index)
            self.assertIn("Smooth_LASSO_rho0_1", summary.index)
            self.assertIn("EW", summary.index)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["grid_edges"], 176)


if __name__ == "__main__":
    unittest.main()
