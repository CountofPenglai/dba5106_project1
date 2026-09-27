"""Frozen current cost-aware LP on the already reviewed July 2026 final-test period.

Run from the repository using its Python environment. This script only evaluates
the LP_Current specification and EW; no selection or parameter search is called.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.data.returns_loader import load_returns
from portfolio.decisions.linear_program_allocation import LPConstraints, LPObjective
from portfolio.engine.forecast_lp_workflow import _prepare
from portfolio.engine.lp_risk_tuning_workflow import LPRiskTuningConfig, _forecasts, _rollout, _summary
from portfolio.evaluation.artifact_store import artifact_path, ensure_fresh_output, RunArtifacts
from portfolio.evaluation.experiment_result import ExperimentResult
from portfolio.evaluation.provenance import code_hashes, file_digest, frame_digest, package_versions
from portfolio.evaluation.result_export import default_code_paths, export_result
from portfolio.paths import PROJECT_ROOT, DEFAULT_CSV


START, END = '2026-07-01', '2026-07-31'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=PROJECT_ROOT/'results/lasso_forecast_lp_stability/20260924T075554Z')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_fresh_output(args.output)
    reference_file = artifact_path(args.reference, 'frozen_decisions.json')
    frozen = json.loads(reference_file.read_text())
    reference_metadata = json.loads(artifact_path(args.reference, 'run_metadata.json').read_text())
    if frozen.get('kind') != 'lp_stability_comparison' or frozen.get('schema_version') != 1:
        raise ValueError('Need the frozen three-version stability experiment')
    periods = LPRiskTuningConfig(**frozen['config']['periods'])
    periods.validate()
    constraints = LPConstraints(**frozen['base_constraints'])
    objective = LPObjective(transaction_cost=periods.transaction_cost)
    if (periods.window != 252 or periods.validation_end != '2026-06-30'
            or constraints != LPConstraints() or periods.transaction_cost != .0005
            or frozen['forecast_alpha'] != 0.002511886431509582):
        raise ValueError('Frozen settings differ from the requested LP_Current specification')
    for name, digest in code_hashes().items():
        if reference_metadata['code_sha256'].get(str(PROJECT_ROOT/name)) != digest:
            raise ValueError('Source differs from the frozen validation run: '+name)
    versions = package_versions()
    for package in ('numpy', 'pandas', 'scikit-learn', 'scipy'):
        if versions[package] != reference_metadata['packages'][package]:
            raise ValueError('Numerical package differs: '+package)
    used = _prepare(load_returns(args.csv, start=periods.history_start, end=END), periods.history_start, END)
    if list(used.columns) != frozen['assets'] or used.shape[1] != 98:
        raise ValueError('Asset universe/order differs')
    if frame_digest(used.loc[:periods.validation_end]) != reference_metadata['used_data_digest']:
        raise ValueError('Pre-July history differs from the frozen validation run')
    # Fixed dates of the established July test: Independence Day is observed July 3.
    expected_dates = pd.bdate_range(START, END).difference(pd.DatetimeIndex(['2026-07-03']))
    dates = used.loc[START:END].index
    if not dates.equals(expected_dates) or len(dates) != 22:
        raise ValueError('July test must contain all 22 established trading dates')
    selection = dict(schema_version=1, kind='frozen_current_lp_july_evaluation',
        model='LP_Current', forecast_alpha=frozen['forecast_alpha'],
        forecast_window=periods.window, risk_window=periods.window,
        objective=asdict(objective), constraints=asdict(constraints),
        reference=str(args.reference.resolve()), reference_sha256=file_digest(reference_file),
        start=START, end=END, assets=list(used.columns),
        configuration_frozen_before_this_run=True, parameter_search=False,
        test_period_previously_reviewed=True, independent_untouched_test=False,
        initialization='Reset to EW on July 1; initial rebalance charged',
        cost_convention='one-way rate times full L1 trades; first-order net=gross-cost; gross holdings drift')
    # Lock the evaluated specification on disk before any July forecast/solve.
    lock = args.output.parent/(args.output.name+'_selection.json')
    with lock.open('x') as file:
        json.dump(selection, file, indent=2)
        file.write('\n')
    notify = lambda message: print(message, flush=True)
    notify('Evaluating frozen LP_Current and EW on July 1–31; no parameter search.')
    predictions, diagnostics = _forecasts(used, periods, frozen['forecast_alpha'], START, END, notify)
    frames = _rollout(used, predictions, dict(LP_Current=objective, EW=None), periods, constraints, notify)
    frames['forecast_predictions'] = predictions
    frames['forecast_diagnostics'] = diagnostics
    frames['final_test_summary'] = _summary(frames, ['LP_Current', 'EW'])
    validation = pd.read_csv(artifact_path(args.reference, 'validation_summary.csv'), index_col=0)
    comparison = []
    for phase, table in [('validation', validation.loc[['LP_Current','EW']]),
                         ('july_final_test', frames['final_test_summary'])]:
        for model, row in table.iterrows():
            values = {key: row[key] for key in frames['final_test_summary'].columns}
            comparison.append(dict(phase=phase, model=model, **values))
    frames['validation_final_comparison'] = pd.DataFrame(comparison).set_index(['phase','model'])
    metadata = dict(selection, phase='additional_july_final_test_evaluation',
        final_test_evaluated=True, alpha_reselected=False,
        evaluation_days=len(dates), used_data_digest=frame_digest(used),
        pre_july_data_digest=frame_digest(used.loc[:periods.validation_end]),
        reference_metadata_sha256=file_digest(artifact_path(args.reference,'run_metadata.json')),
        protocol='causal daily refits; only previously observed test days enter later rolling windows',
        evidence_scope='same previously reviewed July period; no new untouched-test claim')
    result = ExperimentResult(frames, metadata, selection)
    export_result(result, args.output, source_csv=args.csv, layout='categorized',
                  code_paths=[*default_code_paths(), Path(__file__).resolve()])
    print(frames['final_test_summary'].to_string(), flush=True)
    print('Saved '+str(args.output.resolve()), flush=True)
    return result


if __name__ == '__main__':
    main()
