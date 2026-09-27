"""One-shot July 2026 final test of six models selected before seeing July performance."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.data.returns_loader import load_returns
from portfolio.data.history_windows import TrainingWindow, rolling_windows
from portfolio.engine.rolling_backtest import BacktestResult, backtest
from portfolio.evaluation.portfolio_metrics import backtest_metrics
from portfolio.evaluation.provenance import code_hashes, file_digest, frame_digest, package_versions
from portfolio.models.portfolio_variants import ExperimentConstraints, ExperimentLoss, fit_portfolio_variant
from portfolio.models.market_beta_portfolio import fit_portfolio as fit_beta
from portfolio.models.lasso_return_forecast import build_supervised, forecast_at
from portfolio.decisions.linear_program_allocation import LPConstraints, build_lp_context, solve_linear_portfolio
from portfolio.config import TrainingConfig
from portfolio.regularization import Regularization


ROOT = Path('/Users/zhengce/projects/fundation_ba/project1')
SOURCE = ROOT / '100_Portfolios_10x10_Daily.csv'
LEADERBOARD = ROOT / 'docs/experiments/unified-252-expanded-2026-09-23.csv'
FROZEN_LP = ROOT / 'results/unified_252/20260922T122949.119118Z/lp/config/frozen_decisions.json'
OUT = Path('/private/tmp/final_test_july_2026_20260923')
START, END, WINDOW = '2026-07-01', '2026-07-31', 252
SELECTION = {
    'EW': 'baselines/EW',
    'LASSO_Selected': 'loss_search/LASSO_Selected',
    'LASSO_WeightCap_10pct': 'concentration/LASSO_WeightCap_10pct',
    'Ridge_conservative_0.5_0.9': 'market_beta/Ridge_conservative_0.5_0.9',
    'LASSO_market_like_0.8_1.2': 'market_beta/LASSO_market_like_0.8_1.2',
    'LP_LASSO_252d': 'lp/LP_LASSO_252d',
}


def main():
    if OUT.exists():
        raise FileExistsError(f'Output already exists: {OUT}')
    reference = pd.read_csv(LEADERBOARD).set_index('key')
    assert set(SELECTION.values()) <= set(reference.index)
    source_sha = file_digest(SOURCE)
    assert source_sha == '11399a830c7b2a192d652b39e866a74c05d7380dd6edf1573ce1c40f9d271c23'
    returns = load_returns(SOURCE, start='2021-08-01', end=END)
    dates = returns.loc[START:END].index
    assert returns.shape[1] == 98 and len(dates) == 22
    assert dates[0] == pd.Timestamp(START) and dates[-1] == pd.Timestamp(END)
    assert np.isfinite(returns.to_numpy()).all()
    assert (returns.loc[START:END] != -99).to_numpy().all()

    OUT.mkdir(parents=True)
    (OUT / 'weights').mkdir()
    (OUT / 'returns').mkdir()
    (OUT / 'diagnostics').mkdir()
    (OUT / 'selection.json').write_text(json.dumps({
        'selected_before_holdout': True,
        'selection_source': str(LEADERBOARD.relative_to(ROOT)),
        'selection_rule': 'Validation daily Sharpe within each idea; EW, LP near-EW return, and selected-LASSO control retained',
        'holdout_role': 'July 2026 final test, designated by the user before execution',
        'window': WINDOW, 'start': START, 'end': END,
        'models': SELECTION,
        'validation_metrics': {name: {
            key: float(reference.loc[source_key, key])
            for key in ('cumulative_return', 'sharpe', 'volatility', 'max_drawdown', 'average_gross_leverage')
        } for name, source_key in SELECTION.items()},
    }, indent=2) + '\n')

    rows = {}
    checks = {}

    def save(name, result, diagnostics=None):
        if not result.returns.index.equals(dates) or not result.weights.index.equals(dates):
            raise ValueError(f'{name}: wrong evaluation dates')
        weights = result.weights
        actual = returns.loc[dates]
        reconstructed = (weights * actual).sum(axis=1)
        error = float(np.abs(reconstructed.to_numpy() - result.returns.to_numpy()).max())
        sum_error = float(np.abs(weights.sum(axis=1) - 1).max())
        if error > 1e-12 or sum_error > 1e-6:
            raise ValueError(f'{name}: reconstruction or budget failed')
        if diagnostics is not None:
            if not diagnostics.index.equals(dates):
                raise ValueError(f'{name}: diagnostics dates differ')
            if not (pd.to_datetime(diagnostics['history_end']).to_numpy() < dates.to_numpy()).all():
                raise ValueError(f'{name}: future entered historical fit')
            if not (diagnostics['history_count'] == WINDOW).all():
                raise ValueError(f'{name}: history window differs')
            diagnostics.to_csv(OUT / 'diagnostics' / f'{name}.csv')
        metrics = backtest_metrics(result)
        w = weights.to_numpy()
        r = actual.to_numpy()
        previous = np.empty_like(w)
        previous[0] = 1 / w.shape[1]
        previous[1:] = w[:-1] * (1 + r[:-1]) / (1 + result.returns.to_numpy()[:-1, None])
        metrics['average_daily_turnover'] = float((.5 * np.abs(w - previous).sum(axis=1)).mean())
        rows[name] = metrics
        checks[name] = {'max_daily_return_reconstruction_error': error,
                        'max_full_investment_error': sum_error,
                        'max_abs_position': float(np.abs(w).max())}
        weights.to_csv(OUT / 'weights' / f'{name}.csv')
        result.returns.to_frame().to_csv(OUT / 'returns' / f'{name}.csv')
        print(name, 'Sharpe', f"{metrics['sharpe']:.6f}", 'return', f"{metrics['cumulative_return']:.4%}", flush=True)

    def standard(name, strategy):
        records = []
        def observed(history):
            weights, detail = strategy(history)
            records.append({'history_start': history.index[0], 'history_end': history.index[-1],
                            'history_count': len(history), **detail})
            return weights
        result = backtest(returns, observed, start=START, end=END, window=WINDOW)
        save(name, result, pd.DataFrame(records, index=dates))

    standard('EW', lambda history: (pd.Series(1 / len(history.columns), index=history.columns), {}))
    reg = Regularization('lasso', 1e-5)
    loss = ExperimentLoss(reward_strength=0, exposure_strength=2)
    for name, cap in [('LASSO_Selected', None), ('LASSO_WeightCap_10pct', .10)]:
        def strategy(history, cap=cap):
            fitted = fit_portfolio_variant(history, regularization=reg, loss=loss,
                                           constraints=ExperimentConstraints(weight_limit=cap))
            return fitted.weights, {'gross_leverage': fitted.diagnostics['gross_leverage'],
                                    'fit_risk': fitted.diagnostics['variance'],
                                    'max_abs_weight': float(fitted.weights.abs().max())}
        standard(name, strategy)

    for name, kind, bounds in [
        ('Ridge_conservative_0.5_0.9', 'Ridge', (.5, .9)),
        ('LASSO_market_like_0.8_1.2', 'LASSO', (.8, 1.2)),
    ]:
        def strategy(history, kind=kind, bounds=bounds):
            values, beta = fit_beta(history, kind, bounds)
            values = pd.Series(values, index=history.columns)
            gross = float(values.abs().sum())
            portfolio = history.to_numpy() @ values.to_numpy()
            equal = history.mean(axis=1).to_numpy()
            ratio = float(np.var(portfolio) / np.var(equal))
            if gross > 1.50001 or ratio > 1.00001 or not bounds[0] - 1e-5 <= beta <= bounds[1] + 1e-5:
                raise ValueError(f'{name}: market-beta/risk constraint failed')
            return values, {'fitted_beta': beta, 'gross_leverage': gross, 'fit_variance_to_EW_ratio': ratio}
        standard(name, strategy)

    frozen = json.loads(FROZEN_LP.read_text())
    current_codes = code_hashes()
    if any(current_codes.get(path) != digest for path, digest in frozen['code_sha256'].items()):
        raise ValueError('Frozen LP source changed')
    if list(returns.columns) != frozen['assets']:
        raise ValueError('Frozen LP asset order changed')
    if frame_digest(returns.loc[:frozen['config']['validation_end']]) != frozen['development_data_digest']:
        raise ValueError('Frozen LP development data changed')
    config = TrainingConfig(TrainingWindow(**frozen['forecast_configs']['252d']['window']), None)
    risk_window = TrainingWindow(**frozen['risk_window'])
    assert config.window.observations == risk_window.observations == WINDOW
    alpha = frozen['alphas']['252d']
    constraints = LPConstraints(**frozen['lp_constraints'])
    data = build_supervised(returns)
    lp_weights, lp_records = [], []
    for date, history in rolling_windows(returns, start=START, end=END, window=risk_window):
        predicted, forecast = forecast_at(data, date, config, alpha)
        if forecast['train_count'] != WINDOW or forecast['train_end'] >= date:
            raise ValueError('LP forecast training leaked into holdout day')
        context = build_lp_context(history, constraints)
        values, detail = solve_linear_portfolio(context, predicted.to_numpy())
        lp_weights.append(values)
        lp_records.append({'history_start': history.index[0], 'history_end': history.index[-1],
                           'history_count': len(history), 'forecast_train_start': forecast['train_start'],
                           'forecast_train_end': forecast['train_end'], 'forecast_train_count': forecast['train_count'],
                           'alpha': alpha, **detail})
        print('LP', date.date(), flush=True)
    lp_w = pd.DataFrame(lp_weights, index=dates, columns=returns.columns)
    lp_r = pd.Series((lp_w * returns.loc[dates]).sum(axis=1), index=dates, name='return')
    save('LP_LASSO_252d', BacktestResult(lp_r, lp_w), pd.DataFrame(lp_records, index=dates))

    table = pd.DataFrame.from_dict(rows, orient='index').loc[list(SELECTION)]
    table.index.name = 'model'
    table.to_csv(OUT / 'final_test_summary.csv')
    with (OUT / 'verification.json').open('w') as file:
        json.dump({'models': checks, 'evaluation_days': len(dates), 'assets': len(returns.columns),
                   'july_final_test_complete': True}, file, indent=2)
    (OUT / 'run_metadata.json').write_text(json.dumps({
        'source_csv': str(SOURCE), 'source_sha256': source_sha,
        'leaderboard_sha256': file_digest(LEADERBOARD),
        'frozen_lp_sha256': file_digest(FROZEN_LP),
        'frozen_lp_alpha': alpha,
        'frozen_lp_added_nonmatching_source_paths': sorted(set(current_codes) - set(frozen['code_sha256'])),
        'code_sha256': current_codes, 'packages': package_versions(),
        'start': START, 'end': END, 'window': WINDOW,
        'metric_convention': 'daily mean/sample std ddof=1; compounded cumulative return; initial wealth 1 drawdown; no transaction costs',
        'selection_frozen_before_final_test': True,
        'test_period_decision': 'July 2026 designated as final test by user on 2026-09-23; August data unavailable',
    }, indent=2) + '\n')
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
