"""Create a compact report and figures from independently checked results."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/project1-unified252/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

BASE=Path('/private/tmp/project1-unified252')
ROOT=Path((BASE/'current_run.txt').read_text())
OUT=BASE/'analysis'
df=pd.read_csv(OUT/'leaderboard_by_sharpe.csv',index_col=0).fillna('')
daily=pd.read_csv(OUT/'validation_returns.csv',index_col=0,parse_dates=True)
audit=json.loads((OUT/'verification.json').read_text())
loss=json.loads((ROOT/'loss_search/config/run_metadata.json').read_text())
recency=json.loads((ROOT/'recency/config/selection.json').read_text())
frozen=json.loads((ROOT/'mixture/config/frozen_decisions.json').read_text())
families=dict(baselines='基础回归与固定约束',loss_search='收益奖励与敞口惩罚搜索',recency='近期样本加权',elastic_net='Elastic Net',pca='滚动 PCA',market_beta='市场 Beta 约束',mixture='二阶段混合决策',lp='二阶段直接 LP')
representatives=[
 ('baselines/EW','EW'),
 ('lp/LP_LASSO_252d','LASSO + LP'),
 ('market_beta/LASSO_market_like_0.8_1.2','LASSO + Beta [0.8,1.2]'),
 ('loss_search/LASSO_Selected','LASSO + selected exposure penalty'),
 ('market_beta/Ridge_conservative_0.5_0.9','Ridge + Beta [0.5,0.9]'),
 ('mixture/LASSO_252d_Sharpe','LASSO + Sharpe mixture'),
 ('baselines/LASSO_Constrained','LASSO + hard constraints'),
 ('baselines/LASSO','LASSO'),
]
colors=['#202938','#007c91','#d18612','#7857a4','#d94b57','#32875d','#5872ba','#96989d']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
for (key,label),color in zip(representatives,colors):
    r=daily[key]
    dates=pd.DatetimeIndex([pd.Timestamp('2025-12-31'),*r.index])
    wealth=np.r_[1.,np.cumprod(1+r)]
    axes[0].plot(dates,wealth-1,label=label,color=color,lw=2 if key in ('baselines/EW','loss_search/LASSO_Selected','lp/LP_LASSO_252d') else 1.4)
    axes[1].plot(dates,wealth/np.maximum.accumulate(wealth)-1,color=color,lw=1.5)
for ax in axes:
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.16)
axes[0].set_ylabel('Cumulative return')
axes[1].set_ylabel('Drawdown from running peak')
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[0].legend(loc='upper left',ncol=2,fontsize=9,frameon=False)
fig.suptitle('252-observation comparison on the same validation period',x=.08,ha='left',fontsize=16,fontweight='bold')
fig.text(.08,.928,'2 Jan–30 Jun 2026 · 98 assets · gross returns before transaction costs',fontsize=11,color='#555555')
fig.subplots_adjust(left=.08,right=.98,top=.90,bottom=.10,hspace=.12)
fig.text(.08,.025,'Exposure penalty selected on this validation set. Curves describe development performance.',fontsize=10,color='#555555')
fig.savefig(OUT/'unified-252-2026-09-22.png',dpi=180)
plt.close(fig)

def table(keys):
    lines=['| 模型 | 累计收益 | 日 Sharpe | 日波动 | 最大回撤 | 平均总敞口 |','|---|---:|---:|---:|---:|---:|']
    for key in keys:
        row=df.loc[key]
        label=row['model']
        if row['selection'] in ('validation_selected','selected_parameter_ablation'):label+=' †'
        lines.append(f"| {label} | {row.cumulative_return:.2%} | {row.sharpe:.4f} | {row.volatility:.4%} | {row.max_drawdown:.2%} | {row.average_gross_leverage:.3f} |")
    return '\n'.join(lines)

top_return=df.cumulative_return.idxmax()
top_sharpe=df.sharpe.idxmax()
text=f'''# 全模型 252 天统一窗口比较

运行日期：2026-09-22。八类现有实验全部重新运行，共 {len(df)} 个结果条目，包含重复基准、候选组件和选参后别名；按照全期持仓逐元素差异小于 1e-10 去重后为 {audit['unique_holdings']} 组数值不同的持仓。去重数量不是独立建模思想的数量，近似相同的凸优化解仍可能分别计数。

## 统一设置与比较边界

- 验证区间：**2026-01-02 至 2026-06-30，123 个交易日**，98 个资产。预处理首先删除 `ME9 BM10` 和 `BIG HiBM`；本轮涉及的 2021-08-02 至 2026-06-30 面板完整，无缺失值。
- 所有滚动回归、预测拟合和风险估计均使用此前 **252 个交易观测**。二阶段混合决策层另用 **2024-12-30 至 2025-12-31 的 252 个交易日**拟合组合系数；验证阶段不更新这些系数。LP 每日直接优化持仓，不需要独立训练决策系数。
- 原 126 天、三年、五年探索的配置和代码均保留。本轮将相应思路的窗口统一为 252 后比较；等权 LASSO、等权约束 LASSO 等相同配置只需作为同一基准运行。近期加权仍比较半衰期 126/252/504；半衰期是衰减速度，不是历史窗口长度。
- 预测模型的 252 指完整的监督训练目标数；最长 63 天滞后特征需要更早的原始数据作预热。预测 alpha 保留原先截至 2024-07-31 的 3×21 顺序验证；每折训练目标数仍为 252。历史校准期没有移入当前验证期。
- 只统一时间设置，保留各方法已有正则化强度、约束、搜索空间与求解策略。旧 `LASSO_CV/Ridge_CV` 保留五折 CV；前者按 MSE、后者按 RidgeCV 的默认 R² 评分。本轮未将它们改成时间顺序 CV。PCA Ridge 保留历史实现的 alpha 缩放，与核心 Ridge 的缩放不同。
- 日 Sharpe = 日收益均值 / 样本标准差（ddof=1），无风险收益设为零；累计收益按每日复利计算；最大回撤包含初始净值 1，表中以负数表示。全部未扣交易成本。换手率统一按漂移后的持仓计算，验证首日统一从 EW 起算。
- 这段验证数据已用于多轮开发和比较。† 表示本轮直接利用该验证期选参或使用所选参数作消融，不能将这些排名当作独立测试成绩。所有候选中挑出的“最好”也有事后选择偏差。八月最终测试未运行。

## 代表结果

{table([key for key,_ in representatives])}

![统一窗口下的累计收益和回撤](unified-252-2026-09-22.png)

累计收益最高的条目为 `{df.loc[top_return,'model']}`（{df.loc[top_return,'cumulative_return']:.2%}）；日 Sharpe 最高的条目为 `{df.loc[top_sharpe,'model']}`（{df.loc[top_sharpe,'sharpe']:.4f}）。收益、回撤、波动和杠杆没有由同一方法同时占优，不合成为人为的综合评分。

## 结果解读

1. **累计收益：EW 仍领先，LASSO + LP 最接近。** LP 的收益接近 EW，但日波动、回撤、杠杆和换手率均需一起考察；这轮结果没有显示它全面超过 EW。LP 中的回撤限制是训练历史上的非复利累计损益回撤代理，并不直接约束未来复利最大回撤。
2. **较高 Sharpe：保守 Beta Ridge 与软敞口惩罚 LASSO 表现突出。** 前者在较低日波动下取得较高 Sharpe；后者在收益、回撤和总敞口之间取得较好的本期组合。二者的回撤和杠杆结构不同，不能用 Sharpe 单独概括。
3. **收益奖励没有被本轮搜索保留。** 本轮最终 eta={loss['selected_parameters']['reward_strength']:g}、gamma={loss['selected_parameters']['exposure_strength']:g}；因此联合模型与仅敞口惩罚消融相同。搜索停止原因为 `{loss['validation']['stop_reason']}`，评价 {loss['distinct_search_configurations']} 组不同配置；不能把效果归因于未启用的收益奖励。
4. **硬约束仍优于普通 LASSO 的本期结果。** 普通 LASSO 的平均总敞口明显高于 1.5；加入硬约束后累计收益与 Sharpe 上升，最大回撤有所改善。日波动没有明显下降，改善不是每个风险指标都变好。
5. **Elastic Net 和 PCA 的增益有限。** Elastic Net 最好的比例接近纯 LASSO，本期小幅差距不足以说明稳定优势。PCA 最好的候选仍低于普通 LASSO 的收益和 Sharpe，扩大主成分数也没有单调改善。
6. **混合决策层主要选择最低方差组件。** LASSO 预测版本的 Sharpe 和收益减波动模型几乎都把权重给了 V，因此其表现不能作为预测信号有效的证据。历史均值预测的收益减波动版本更偏向 U，收益略高，但波动和回撤明显更大。
7. **近期加权没有超过等权约束基准。** 本轮在三种半衰期中选出 {recency['selected_half_life']} 个交易日，累计收益 {df.loc['recency/LASSO_252d_Weighted_Selected','cumulative_return']:.2%}、日 Sharpe {df.loc['recency/LASSO_252d_Weighted_Selected','sharpe']:.4f}，仍低于等权约束 LASSO 的 8.80% 和 0.1223。选择依据仍是当前验证期累计收益，选中模型不构成额外的独立实验。

## 全部结果

以下包括所有 65 个条目；不同实验中重复出现的 EW、LASSO 和消融对照故意保留，以便复查各实验。CSV 的 `equivalent_holdings` 指向持仓数值相同的先前条目。
'''
for family,title in families.items():
    subset=df[df.family==family].sort_values('sharpe',ascending=False)
    text+=f'\n### {title}\n\n'+table(subset.index)+'\n'
    if family=='market_beta':text+='\n此处 `unconstrained` 仅指没有额外 Beta 上下界，仍有总敞口、历史方差和财富底线约束。Beta 相对同窗口的 EW 收益估计，不是外部市场指数。\n'
    if family=='mixture':text+='\nU 为预测收益最大化候选，V 为收缩协方差下的最低方差候选，E 为 EW。组件属于对照，不是额外训练出的混合模型。\n'
text+=f'''
## 独立核验

- 八个进程全部成功完成；进一步检查了参数搜索的比较状态与失败记录，未用进程退出码替代模型级核验。
- 每个条目均对齐同样的 123 个日期和 98 列资产。用原始资产收益与导出持仓重新计算每日收益和全部核心指标，最大收益重构误差 {audit['max_return_reconstruction_error']:.3g}，最大指标误差 {audit['max_metric_reconstruction_error']:.3g}，最大持仓和偏差 {audit['max_budget_error']:.3g}。
- 对导出的训练计数和历史截止日进行核验；对约束模型逐日重新计算训练风险、总敞口、财富底线、Beta 上下界及 LP 单资产/MAD/历史回撤限制。各项在既定数值容差内通过。市场 Beta 历史求解器允许 optimal_inaccurate，本轮没有依靠该标记替代实际可行性检查。
- 数据哈希与八类运行时记录一致，代码哈希与当前源文件一致。使用的是当前已重构、尚未提交的工作区代码，因此仅凭 HEAD 不能完整复现；元数据保存了逐文件源码哈希。
- 独立 agent 审查了方法覆盖、252 观测设置和信息时序，并指出换手率起点差异；统一汇总已修正此口径，未修改原始实验产物。

数据 SHA-256：`{audit['data_sha256']}`。

## 保存位置与复现

- 本报告的完整数值表：[unified-252-2026-09-22.csv](unified-252-2026-09-22.csv)。指标数值保存为小数，百分比仅用于展示。
- 本次运行根目录：`results/unified_252/{ROOT.name}/`。
- 根目录 `metrics/` 包含按 Sharpe / 累计收益排序的总表、去重表及分组摘要；`returns/validation_returns.csv` 包含统一验证期每日收益；`diagnostics/reconstruction_checks.csv` 包含每个条目的独立核验。
- 每个实验族子目录保留 `metrics/returns/weights/predictions/diagnostics/config/search` 中实际产生的内容和清单；根目录 `logs/` 保存完整日志，`config/` 保存状态、数据审计和本轮脚本。
- `turnover-252.md`、`smoothness-252.md`、`shrinkage-252.md` 对应的独立运行代码当前缺失，故没有将历史报告数字混入本轮排名。二阶段风险估计中已有的 Ledoit-Wolf 收缩正常运行。原教学脚本归档不是额外维护模型，其基础方法已被框架覆盖。
- 原来的 126/252 天与多年配置、历史结果目录均未覆盖或删除。本轮没有修改模型算法，也没有提交或推送。

用本次保存的 runner 为指定实验族创建一个全新的结果目录，例如：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
.venv/bin/python results/unified_252/{ROOT.name}/config/run_family.py \\
  --project "$PWD" --family lp --output results/unified_252/lp_reproduction
```

可用实验族：`baselines`、`loss_search`、`recency`、`elastic_net`、`pca`、`market_beta`、`mixture`、`lp`。需保留本轮元数据中的相同数据和源码版本，复现目录必须为空或尚不存在。原始大型产物按仓库设置位于忽略的 `results/`；报告、简表和图片可单独纳入后续提交。
'''
(OUT/'unified-252-2026-09-22.md').write_text(text)
print('Created report and figure in',OUT)
