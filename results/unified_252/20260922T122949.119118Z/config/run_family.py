"""Reproduce one family using existing project APIs and a uniform 252-day setup."""
import argparse
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--family',choices=('baselines','elastic_net','pca','market_beta','loss_search','recency','mixture','lp'),required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.project))
import numpy as np
import pandas as pd
from portfolio.data.returns_loader import load_returns
from portfolio.config import TRAINING_PRESETS, TrainingConfig
from portfolio.data.history_windows import TrainingWindow
from portfolio.evaluation.artifact_store import RunArtifacts
from portfolio.evaluation.provenance import run_provenance, file_digest
from portfolio.evaluation.result_export import export_result

CSV=args.project/'100_Portfolios_10x10_Daily.csv'
START,END,WINDOW='2026-01-02','2026-06-30',252
risk=TrainingWindow(observations=WINDOW)
base=['--csv',str(CSV),'--output',str(args.output),'--layout','categorized']
print('Starting',args.family,'with all lookbacks =',WINDOW,flush=True)
np.random.seed(0)
cli={
 'baselines':('regularized_baselines',['--window',str(WINDOW),'--start',START,'--end',END,'--weighted','--growth']),
 'elastic_net':('elastic_net_portfolio',['--window',str(WINDOW),'--start',START,'--end',END]),
 'pca':('pca_regularized_portfolio',['--window',str(WINDOW),'--start',START,'--end',END]),
 'market_beta':('market_beta_constraints',['--window',str(WINDOW),'--start',START,'--end',END]),
 'loss_search':('lasso_reward_exposure_search',['--window',str(WINDOW),'--validation-start',START,'--validation-end',END]),
}
if args.family in cli:
 module,options=cli[args.family]
 importlib.import_module('experiments.'+module).main(base+options)
else:
 returns=load_returns(CSV,start='2021-08-01',end=END)
 if not np.isfinite(returns.to_numpy()).all() or returns.shape[1]!=98:
  raise ValueError('Expected a complete panel of the same 98 assets')
 if args.family=='recency':
  from portfolio.constraints import PortfolioConstraints
  from portfolio.engine.lasso_window_strategy import lasso_strategy
  from portfolio.engine.rolling_backtest import backtest
  from portfolio.evaluation.portfolio_metrics import backtest_metrics
  from experiments.lasso_recency_weight_search import select_half_life
  store=RunArtifacts(args.output,'categorized').prepare()
  rows, scores, fitted={}, {}, {}
  variants=[('LASSO_252d_Weighted_HL252',TrainingConfig(risk,half_life=252),None)]
  variants += [(f'LASSO_252d_HL{half}_Constrained',TrainingConfig(risk,half_life=half),PortfolioConstraints()) for half in (126,252,504)]
  for name,config,constraints in variants:
   print('Running',name,flush=True)
   records=[]
   result=backtest(returns,lasso_strategy(config,records,constraints=constraints),start=START,end=END,window=risk)
   diagnostics=pd.DataFrame(records,index=result.returns.index)
   if not (diagnostics.observations==WINDOW).all():raise ValueError('Non-252 training window')
   store.write_backtest(name,result,diagnostics)
   rows[name]=backtest_metrics(result)
   if constraints is not None:
    scores[config.half_life]=rows[name]['cumulative_return'];fitted[config.half_life]=(result,diagnostics)
  selected=select_half_life(scores)
  chosen='LASSO_252d_Weighted_Selected'
  result,diagnostics=fitted[selected]
  store.write_backtest(chosen,result,diagnostics)
  rows[chosen]=backtest_metrics(result)
  pd.DataFrame(rows).T.to_csv(store.path('summary.csv'))
  store.path('selection.json').write_text(json.dumps(dict(selected_half_life=selected,scores=scores,selection_metric='validation cumulative return'),indent=2))
  meta=dict(**run_provenance(CSV),window=WINDOW,validation_start=START,validation_end=END,
            training_settings={name:asdict(config) for name,config,_ in variants},
            selected_half_life=selected,half_life_candidates=[126,252,504],
            artifact_layout='categorized',artifact_layout_version=1,
            evaluation_role='validation_used_for_half_life_selection',final_test_evaluated=False)
  store.path('run_metadata.json').write_text(json.dumps(meta,indent=2))
  store.finalize()
 else:
  end='2025-12-31'
  decision_days=returns.loc[:end].index[-WINDOW:]
  decision_start=str(decision_days[0].date())
  configs={'252d':TRAINING_PRESETS['252d']}
  if args.family=='mixture':
   from portfolio.engine.forecast_mixture_workflow import ExperimentConfig,run_experiment
   config=replace(ExperimentConfig(),decision_start=decision_start,decision_end=end)
   print('Mixture decision training:',decision_start,'to',end,'observations:',len(decision_days),flush=True)
   result=run_experiment(returns,config=config,forecast_configs=configs,risk_window=risk,
                         include_baselines=False,progress=lambda msg:print(msg,flush=True))
   if result.metadata['decision_training_days']!=WINDOW:raise ValueError('Decision layer must use 252 observations')
  else:
   from portfolio.engine.forecast_lp_workflow import DirectLPConfig,run_direct_lp
   config=replace(DirectLPConfig(),development_start=decision_start,development_end=end)
   result=run_direct_lp(returns,config=config,forecast_configs=configs,risk_window=risk,
                        include_development=False,progress=lambda msg:print(msg,flush=True))
  result.metadata['benchmark_window']=WINDOW
  result.metadata['benchmark_configuration']='252 forecast observations; 252 risk observations; 252 mixture-training observations when applicable'
  export_result(result,args.output,source_csv=CSV,layout='categorized')
# Save the exact runner identity separately without changing model outputs.
store=RunArtifacts(args.output,'categorized')
store.path('benchmark_runner.json').write_text(json.dumps(dict(family=args.family,window=WINDOW,
 validation_start=START,validation_end=END,runner_sha256=file_digest(__file__),random_seed=0),indent=2))
store.finalize()
print('Completed',args.family,flush=True)
