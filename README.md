# DBA5106 Portfolio Training Framework

一个以模块封装为重点的线性组合训练框架。数据、分割、模型、损失、正则化、样本权重、组合权重和评价相互独立，使用普通函数连接。

## 安装与运行

需要 Python 3.10+。在项目目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m experiments.regularized_baselines --weighted
python -m experiments.regularized_baselines --weighted --growth
python -m experiments.lasso_training_windows --presets 252d 3y --start 2026-01-02
python -m experiments.lasso_reward_exposure_search
python -m unittest discover -s tests -v
```

CSV 默认放在项目根目录，也可以用 `--csv /path/to/file.csv` 指定。`--start`、`--end` 和 `--window` 控制滚动评价区间和历史窗口。

默认使用 252 个交易日的训练窗口，比较 EW、MinVar、固定强度 LASSO/Ridge，以及每日重新选参的 LASSO_CV/Ridge_CV。`--weighted` 增加 RidgeWeighted、LASSO_CV_Weighted、Ridge_CV_Weighted，均同时使用 63 日半衰期的时间权重和 25% 的 EW 混合。原始作业源码保存在 `examples/original_assignment.py`，`5106_assignement_01.py` 保留为兼容启动器。推荐入口为 `python -m experiments.regularized_baselines`。底层 `backtest`/`rolling_windows` 的默认窗口仍为 126；直接调用它们时请显式传 `window=252`。

## 模块与数据流

`CSV → 收益矩阵 → 历史窗口 → 回归问题 → 线性拟合 → 组合调整 → 样本外评价`

当前目录按数据、模型、决策、执行流程和评估分层；实验入口集中在 `experiments/`。
完整文件职责与所有保留方法见 [架构与方法索引](docs/architecture/portfolio-framework.md)。

| 主要位置 | 职责 |
|---|---|
| `portfolio/data/` | 预处理、时间窗口、训练样本权重 |
| `portfolio/models/` | 最小方差回归、收益预测、约束拟合、Elastic Net/PCA/市场 Beta |
| `portfolio/decisions/` | 候选组合、混合决策和直接 LP |
| `portfolio/engine/` | 拟合组装、逐日执行、参数搜索和求解器调用 |
| `portfolio/evaluation/` | 指标、结果导出、目录规则和可复现性检查 |
| `portfolio/losses.py`、`regularization.py`、`constraints.py` | 损失、正则化和约束定义 |
| `portfolio/config.py` | 126 天、252 天与三年窗口等可复用训练设置 |
| `experiments/` | 每个探索方向的配置与运行入口 |

新入口默认保存到 `results/<实验名>/<UTC运行时间>/`，按 `metrics`、`returns`、`weights`、
`predictions`、`diagnostics`、`config`、`search` 等目录分类，根目录的 `artifact_manifest.json`
记录各文件的位置。使用 `--output` 指定运行根目录、`--layout flat` 切换为平铺格式。
路径规则、读取示例与历史结果说明见 [输出目录说明](docs/output-layout.md)。

原来的 `run_*.py` 和探索脚本保留为兼容入口，默认仍使用平铺格式；下文的旧命令仍然可用。
旧模块导入同样保留，但新增代码应使用上表中的规范位置。

### 一次训练

```python
from functools import partial
from portfolio import Regularization, fit_portfolio, load_returns
from portfolio.data.observation_weights import exponential
from portfolio.decisions.holding_weights import shrink_to_equal

returns = load_returns("100_Portfolios_10x10_Daily.csv")
history = returns.loc[:"2025-12-31"].tail(126)
result = fit_portfolio(
    history,
    regularization=Regularization("ridge", strength=1e-4),
    sample_weights=partial(exponential, half_life=63),
    adjust_weights=partial(shrink_to_equal, strength=0.25),
)
print(result.weights)
print(result.training_loss, result.objective)
```

不传这些参数时使用 OLS、样本等权和原始组合。换成 `Regularization("lasso", 1e-5)` 即为 LASSO。`FitResult` 返回线性模型、原始组合权重、最终组合权重、训练损失和含惩罚的目标值。

训练目标为 `sum(s * (y - prediction)**2) / sum(s) + strength * penalty(beta)`。截距不受惩罚；LASSO 使用 L1 范数，Ridge 使用 L2 范数的平方。适配层统一了两种 sklearn 求解器的缩放约定，因此这里的 `strength` 不能直接当作原始脚本的 sklearn `alpha`。

`X = R @ N`、`y = R @ EW`，其中 `N = [I; -1ᵀ]`，最终 `w_raw = EW - N @ beta`。这里的 y 是历史同期等权组合收益，不是下一日预测标签。保留截距才能将平方误差对应到组合方差。L1 稀疏化 beta 不等于直接筛掉资产。

组合调整为 `w_final = (1-gamma) * w_raw + gamma * EW`，保持资金权重和为 1。允许负权重。`training_loss` 和 `objective` 描述调整前的回归拟合；最终组合表现由回测评价。

### 分割与滚动评价

```python
from portfolio import split_returns, backtest, summarize

parts = split_returns(
    returns,
    train_end="2025-12-31",
    validation_end="2026-06-30",
    test_start="2026-08-01",
    test_end="2026-08-30",
)
strategy = lambda history: fit_portfolio(
    history, regularization=Regularization("ridge", 1e-4)
).weights
validation = backtest(
    returns, strategy,
    start=parts.validation.index.min(), end=parts.validation.index.max(), window=126,
)
print(summarize(validation.returns))
print("Test data available:", not parts.test.empty)
```

`split_returns` 提供互不重叠的固定区间，供单次训练或未来调参使用。滚动评价则每天重新取得此前 126 个交易日；已经发生的验证收益可进入随后日期的训练。`backtest` 使用完整历史矩阵作为输入，但每次只把当前日期之前的窗口交给策略函数。策略函数自身不应从闭包或外部文件读取未来数据。

`rolling_windows` 的起止日期应落在已有数据覆盖范围内；对超出末日的区间会报错，避免静默缩短评价期间。最终测试数据齐备并冻结方案后，才使用测试集的实际首末交易日回测。

现有 CV 策略在每个滚动窗口内使用 5 折和 `np.logspace(-8, 8, 21)`；当前 LassoCV 评分为 MSE，RidgeCV 的 `cv=5, scoring=None` 默认评分为 R²。本次增长实验保留原有评分方式，以保证现有九个策略的比较口径不变。课程要求的 MSE 选参口径需要作为单独变更对齐，不能把当前两者都描述为 MSE。

### 可配置训练窗口与五年历史 LASSO 比较

`run_lasso_windows.py` 只比较固定强度 `1e-5` 的 LASSO。历史数据限定为
2021-08-01 至 2026-07-31，默认在前三年预热后评价 2024-08-01 至 2026-06-30；
另外单独输出原 2026-01-02 至 2026-06-30 验证区间的成绩。
最终 2026-08-01 至 2026-08-30 测试区间保持不变，不由此入口评价。

```bash
# 默认：252 天 baseline、三年等权、三年近期加权
python run_lasso_windows.py --output results/lasso_windows_5y

# 只需改 --presets，就能选择训练配置；126 天依然可用
python run_lasso_windows.py --presets 126d 252d 3y_weighted --output results/lasso_windows_alternative

# 只复现原来的半年验证区间
python run_lasso_windows.py --start 2026-01-02 --end 2026-06-30 --output results/lasso_windows_validation
```

| 预设 | 每次训练窗口 | 样本权重 |
|---|---|---|
| `126d` | 此前 126 个交易观察值 | 等权 |
| `252d` | 此前 252 个交易观察值 | 等权 |
| `3y` | 此前三个日历年 | 等权 |
| `3y_weighted` | 此前三个日历年 | 指数加权，半衰期 252 个交易观察值 |

`TrainingWindow` 只负责窗口边界，`TrainingConfig` 组合窗口与样本权重；模型和回测仍使用原函数。
按年窗口包含三年前对应日期、排除预测当天，实际交易日数可变；不足三年时允许较短历史，
但三年预设至少需要 252 个观察值。默认评价起点已预留完整三年历史。
这项最小样本数是防止意外短窗口的运行保护，不是额外选参。

```python
from portfolio import TRAINING_PRESETS, Regularization, backtest, fit_portfolio

config = TRAINING_PRESETS["3y_weighted"]  # 改这一行即可切换窗口及权重配置

def strategy(history):
    return fit_portfolio(
        history,
        regularization=Regularization("lasso", 1e-5),
        sample_weights=config.sample_weights,
    ).weights

result = backtest(returns, strategy, start="2026-01-02", end="2026-06-30", window=config.window)
```

自定义时可使用 `TrainingConfig(TrainingWindow(years=2, min_observations=252), half_life=252)`；
窗口以日历年为单位，半衰期以交易观察值为单位，二者不混用。
旧的 `backtest(..., window=126/252)`、底层默认 126 天、`run_example.py` 默认 252 天及
`run_tuning.py` 均保留。原有 CV 方法也不受影响；默认固定 LASSO 比较不引入 CV、收益奖励、
总敞口限制或事后 EW 混合。

输出包括每日收益、持仓、训练日期/样本数/有效样本数/目标值，以及数据和代码哈希。
`summary.csv` 对应完整请求区间；仅当完整覆盖原半年验证期时，才另外生成
`validation_summary.csv`。指标直接使用逐日真实组合收益，评价阶段不采用训练样本权重。
已有结果目录不允许覆盖；拟合不收敛会中止运行，不静默接受近似结果。

实验设计、完整结果与解释见 [五年历史 LASSO 窗口实验](docs/experiments/lasso-windows-5y.md)。

使用 `--constrained` 可为每个所选窗口增加原有硬约束版本，并保留同窗口无约束对照：

```bash
python run_lasso_windows.py --presets 252d 3y --constrained --output results/lasso_windows_constrained
```

约束为总敞口 `sum(abs(w)) <= 1.5`、训练波动不超过同窗口/同样本权重的 EW、
所有训练场景财富倍数 `1 + R @ w >= 1e-6`。模型仍固定 LASSO `1e-5`，
显式设置 `reward_strength=0` 和 `exposure_strength=0`，不进行参数搜索。
直接复用 `fit_portfolio_growth` 和原求解器的可行性检查；`_Constrained` 结果额外导出
求解器状态、约束实际值和目标分项，各策略的具体限额记录在元数据中。
三年等权加约束的实验结果见 [三年 LASSO 与原有约束](docs/experiments/lasso-windows-3y-constrained.md)。



### 三年加权约束 LASSO：验证集选择半衰期

`run_weight_tuning.py` 固定三年日历窗口、LASSO `1e-5` 和原硬约束，只调整指数样本权重的半衰期。
默认预先指定 `126 / 252 / 504` 个交易观察值三档，在原 2026-01-02 至 2026-06-30 验证期逐日拟合，
全部候选完成相同日期后选择累计收益最高的一档；收益精确相同时优先较长半衰期。
252 天等权约束版和三年等权约束版作为对照，不参与半衰期选择。

```bash
python run_weight_tuning.py --output results/lasso_3y_weight_tuning
# 修改候选配置即可，无须修改拟合或回测流程
python run_weight_tuning.py --half-lives 126 252 504 --output results/lasso_3y_weight_tuning_alternative
```

继续复用 `TrainingConfig`、现有拟合/回测函数，以及 `portfolio/engine/lasso_window_strategy.py` 的策略适配和公共评估函数。
训练方差和 EW 风险限额使用同一组归一化样本权重；验证收益按真实日收益计算，不使用近期加权。
收益奖励、额外敞口惩罚、EW 混合均关闭，LASSO 强度和约束上限不参与搜索。
候选网格固定，不在看到成绩后扩展搜索；每个候选使用完整验证期，没有逐日挑选当日赢家。

输出 `summary.csv`、每个候选/对照的逐日收益、持仓与约束诊断，以及 `run_metadata.json`。
`selection.json` 包含选中的半衰期、可复用训练设置和模型约束。
旧 `3y_weighted` 预设仍保留半衰期 252，不会被此次选择自动覆盖。
命令只允许原验证区间内的日期，七月和最终八月测试都不进入选择；任一拟合失败会中止，不输出完成的选择结果。

**选中配置在该验证期的成绩具有选择偏差，不能当作独立测试表现。**
最终八月测试仍需在冻结参数并获得完整数据后评价。
详见 [三年加权约束 LASSO 的权重选择实验](docs/experiments/lasso-3y-weight-tuning.md)。

### 固定 LASSO 的对数收益奖励实验

`--growth` 在原有策略之外新增两个可比较的实验：

- `LASSO_Constrained`：固定 LASSO `strength=1e-5`，添加约束，收益奖励 `eta=0`。
- `LASSO_LogGrowth`：相同 LASSO 强度和约束，固定收益奖励 `eta=0.01`。

损失为 `variance(R @ w) - eta * mean(log(1 + R @ w)) + strength * sum(abs(beta))`，保留 `w = EW - N @ beta`。收益奖励使用实际组合收益 `R @ w = y - X @ beta`，不减去回归截距。截距由平均组合收益确定，所以方差项与原来的含截距平方损失一致。模型仍是线性的，求解器改为 CVXPY/Clarabel；只修改损失显示函数不会改变训练。

本次两个实验均使用均匀样本权重，不做 CV、不调 LASSO 强度、不进行事后 EW 混合。默认约束：

- `sum(abs(w)) <= 1.5`，即全额投资组合的总空头最多 25%。
- 训练窗口内的组合波动不超过同一窗口 EW 的波动（训练方差使用归一化样本权重，ddof=0）。
- 每个历史场景的 `1 + R @ w >= 1e-6`，保证对数定义域，不裁剪非法收益。

`--reward-strength`、`--leverage-limit`、`--volatility-limit` 可以显式指定新实验设置，程序不会搜索它们。损失项尺度不同，因此 `eta` 不能与 LASSO 的 `strength` 混为一谈。`eta=0` 只在相同约束下退化为基线；新增约束本身也可能改变原有 LASSO 结果。

```python
from portfolio import fit_portfolio_growth, Regularization, LogGrowthLoss, PortfolioConstraints

fit = fit_portfolio_growth(
    history,
    regularization=Regularization("lasso", 1e-5),
    loss=LogGrowthLoss(reward_strength=0.01),
    constraints=PortfolioConstraints(leverage_limit=1.5, volatility_limit=1.0),
)
print(fit.diagnostics)  # 方差、增长率、奖励、正则项、约束限额和求解状态
```

输出包含每日收益、权重、两个实验的逐日训练诊断、`summary.csv`，以及记录数据/代码 SHA-256、软件版本和全部实验参数的 `run_metadata.json`。诊断按实际持有日对齐，`history_end` 记录训练末日。优先使用 Clarabel；若发生数值失败、精度不足或约束复核不通过，用 SCS 重解同一个目标。仅返回状态为 optimal 且通过实际约束复核的解，`solver` 和 `solver_attempts` 记录求解过程。所有求解均失败则中止，不用 EW 静默替代。

每次导出使用新的或空的目录，已有文件的目录会被拒绝，避免失败运行与旧的 summary/metadata 混在一起。只有全部策略完成后才会生成 summary 和 metadata；缺少它们的目录是未完成的运行。

实验评价仍使用 2026-01-02 至 2026-06-30，主验证指标为负 Sharpe，同时比较累计收益、波动、回撤和实际总敞口。提高训练期对数收益不保证提高未来累计收益；训练波动上限也不保证未来波动不超过 EW。最终 2026-08-01 至 2026-08-30 测试集不参与参数选择。

### 带参数的 LASSO + 约束 + 收益奖励

`run_tuning.py` 使用同一个线性模型，只选择收益奖励强度 eta 和总敞口惩罚强度 gamma。训练窗口默认 252 天，LASSO 强度固定为 `1e-5`。训练目标为：

```text
variance(R @ w) + 1e-5 * sum(abs(beta))
- eta * mean(log1p(R @ w))
+ gamma * variance(R @ EW) * max(0, (sum(abs(w))-1)/(gross_limit-1))**2
```

gamma 默认为零，因此原来的 `run_example.py --growth` 和历史方法保持不变。这里的 gamma 是训练中的总敞口惩罚强度，与旧策略事后向 EW 混合的系数不同。固定总敞口上限 1.5 时，归一化惩罚在总敞口 1、1.25、1.5 分别为 0、0.25、1；乘以训练窗口的 EW 方差用于控制量级。总敞口上限设为 1 时，硬约束已经禁止空头，定义额外惩罚为零，避免除零。三个原有硬约束继续保留。

```python
fit = fit_portfolio_growth(
    history,
    regularization=Regularization("lasso", 1e-5),
    loss=LogGrowthLoss(reward_strength=0.003, exposure_strength=0.1),
    constraints=PortfolioConstraints(),
)
```

`training_loss` 包含方差、收益奖励和敞口惩罚；`objective` 另外包含 LASSO 项。每日诊断单独导出 `exposure_unit_penalty`、`exposure_scale` 和 `exposure_penalty`。

默认选参流程：

1. 在 2025-07-01 至 2025-12-31 的历史开发期评价九组粗网格：eta 为 `0, .003, .01`，gamma 为 `0, .1, 1`。每一天仍使用此前完整 252 天拟合。
2. 在 2026-01-02 至 2026-06-30 验证集重新评价两个旧对照、历史选择及其单项消融配置。历史得分不混入验证排名或停止计数。
3. 围绕当前累计收益最高的配置，对单个正系数减半/加倍；从零出发用对应最小正粗网格值的一半。eta 不超过 .03，gamma 不超过 3。完整评价一个邻域批次后再判断停止，不截短任何候选的回测。
4. 默认最多尝试 15 组不同参数：历史粗网格 9 组，验证期最多额外评价 6 组。相同参数在两个时期的评价分别记录，因此评价次数可超过 15。若剩余预算不足以完成下一批，直接停止。
5. 累计收益改善至少 .002（**0.2 个百分点**）才重置停止计数；默认一轮完整邻域搜索无足够改善就停止。初始批次不受早停影响。比较接近（距最高累计收益不超过 .002）时，依次优先较小 eta、较低实际平均总敞口，最后以较小 gamma 确定同分顺序。

这是**参数搜索早停**，每次凸优化仍正常求解，不是神经网络式 epoch 早停。它限制搜索次数，不能保证全局最优，也不能证明消除了验证集过拟合。eta 和 gamma 可以为零，允许退回已有简单方法。lambda 不参与搜索。

配置冻结后，对比 EW、原始 LASSO、约束版、固定奖励版、所选 eta 的仅奖励版、所选 gamma 的仅惩罚版，以及联合选择版。最后两个消融只用于解释结果，不反馈给搜索。选择指标为验证累计收益，同时报告课程主指标负 Sharpe、波动、回撤和实际总敞口。验证期已经参与选参，必须标记为开发成绩；最终八月测试在冻结方案且数据齐备后独立执行。

入口支持 `--screen-start/--screen-end`、`--validation-start/--validation-end`、`--window`、`--max-candidates`（9–15）、`--min-improvement`、`--patience`。所有日期必须使用完整的 `YYYY-MM-DD` 格式，避免年份或月份字符串扩大数据切片范围。验证结束日期必须早于保留测试期起点 `2026-08-01`；即使手动修改日期，也会在读取数据和训练前拒绝进入测试期或其后的调参区间。后续实验修改这些配置也属于调参，需要如实报告。每次使用新的或空的 `--output` 目录：

```bash
python run_tuning.py --output results/parameterized_lasso_252
```

输出含两个时期的搜索记录、每个成功候选的完整收益/权重/训练诊断、七种比较结果、选中与原始最高分配置、停止原因和运行元数据（代码/数据哈希、软件版本、数据区间）。阶段内的 `candidate_attempts.csv` 每次评价后保存，即使全部初始候选失败而中止也保留错误记录。失败配置不重复求解；若对照或消融无法求解，summary 和 metadata 会标记失败，不用 EW 替代。没有最终 metadata 的输出目录不能视为完成的实验。日历区间中的周末和节假日不产生持仓行，元数据同时记录请求日期与实际首末交易日。

### 扩展方式

- **样本权重**：新增 `weight_fn(n) -> numpy.ndarray`，按历史行顺序返回 n 个有限非负权重，至少两个为正；传给 `sample_weights`。指数衰减按交易观察数计算半衰期。
- **组合调整**：新增 `adjust_fn(raw_weights) -> numpy.ndarray`，维持原资产顺序和权重和为 1；传给 `adjust_weights`。
- **拟合器**：新增 `fit_fn(X, y, *, regularization, sample_weight) -> LinearModel`，传给 `model_fit`。当前训练接口要求拟合器求解统一的平方损失目标。若新增其他损失，需一起扩展求解器和训练损失报告；不能只改损失显示函数。
- **完整策略**：实现 `strategy(history: DataFrame) -> Series`，返回以资产名为索引的权重，然后交给 `backtest`。策略返回的资产顺序可以不同，回测会按名称对齐。

## 数据与评价约定

- 只读取 `Average Equal Weighted Returns -- Daily` 表；预处理入口清理列名首尾空格后，立即按名称删除 `ME9 BM10`（原第 90 列）和 `BIG HiBM`（原第 100 列），再进行日期处理与数值清洗。不依赖列位置，不修改原始 CSV；标准 100 资产数据处理后保留 98 个资产。
- 输入原始值 1 表示 1%，预处理后为 0.01。内部始终使用小数收益；不要重复预处理或再除以 100。
- `-99.99`、`-999` 转为 NaN，保留供检查；训练或回测用到它们会报错，不做隐式填补。
- 样本权重只作用于训练，默认评价不加权。Sharpe 使用日均收益除以样本标准差（`ddof=1`），不扣除无风险收益，不年化。零波动时 Sharpe 为 NaN。
- 累计收益使用复利；最大回撤从初始净值 1 计算，返回负数。回测假设每日按权重配置并持有一天，不计交易费用。
- 数据集会持续更新，读取时按表标题定位，不固定起始行、记录数或最后日期。更新文件后应重新检查实际覆盖日期和保留资产的缺失值；`BIG HiBM` 的七月缺失不会进入后续处理。只有文件已包含完整测试区间，才能进行最终测试；验证成绩不能当作最终测试成绩。

## 验证重点

测试涵盖 CSV 表段与单位、时间边界、解析加权 LASSO/Ridge、协方差 MinVar 等价性、样本权重缩放不变性、组合调整、未来信息隔离以及收益评价。真实数据检查使用文档的六资产 2026-01-02 权重和单日收益。

新损失测试还覆盖独立两资产最优解、收益奖励、三类组合约束、零奖励退化、求解器恢复和完整导出流程。

## 实验记录

- [最新统一 252 天结果](docs/experiments/unified-252-2026-09-22.md)：八类实验、65 条配置与对照，包含约束、收益奖励、敞口惩罚及全部其他探索方向。
- [结果保留清单](docs/results-retention.md)：唯一活动 252 天结果、三年研究例外、缺失的历史窗口结果和清理记录。
- 保留的历史报告（当前缺少可运行实现）：[Turnover](turnover-252.md)、[Grid smoothness](smoothness-252.md)、[Covariance shrinkage](shrinkage-252.md)。它们及配图均保留，但不混入最新重跑成绩。
- [三年窗口与近期加权](docs/experiments/lasso-windows-5y.md)、[三年约束 LASSO](docs/experiments/lasso-windows-3y-constrained.md)、[三年权重选择](docs/experiments/lasso-3y-weight-tuning.md)：保留的三年研究。

最新对比将拟合、预测、风险估计和混合决策训练统一为 252 个交易观测。下文二阶段 CLI 的历史默认值仍使用三年风险窗口，用于保留三年研究方向；这些结果与全 252 天结果分开保存。126 天及三年训练设置继续可用。

## Two-stage LASSO forecasting and fixed portfolio mixtures

`run_two_stage.py` adds a separate forecasting experiment. The existing minimum-variance regression and its 126-day, 252-day, and three-year runners keep their behavior.

The new forecasters use six lagged return/risk features and predict each asset's next daily return. Each forecast window (252 complete trading samples or at most three calendar years) selects one shared LASSO alpha using three chronological 21-day folds ending in July 2024. Alpha then stays fixed while forecast coefficients are refitted daily. Both variants use equally weighted training observations.

A common three-year Ledoit–Wolf risk estimate produces three daily candidate portfolios: maximum predicted return, minimum estimated variance, and equal weight. Each candidate satisfies full investment, gross leverage <= 1.5, estimated volatility <= EW, and the historical wealth floor. The second stage learns a nonnegative mixture of these candidates, either maximizing historical Sharpe or maximizing mean return minus gamma times **standard deviation**. It trains on actual returns from candidates constructed out of sample; it does not train on fitted predictions from the same observations.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_two_stage.py \
  --output results/two_stage_lasso
```

The output directory must be new or empty. The default run uses decision training from August 2024 through December 2025 and development validation from January through June 2026. Mixture weights remain fixed throughout validation. Only gamma is selected from five predeclared candidates using validation cumulative return; all candidate fits are retained. The run includes historical-mean forecast controls, the individual candidate portfolios, and the existing 252-day/three-year LASSO baselines with and without constraints.

Outputs include `comparison.csv`, `validation_summary.csv`, `prediction_validation.csv`, `decision_fits.csv`, `gamma_validation.csv`, per-strategy weights/returns/diagnostics, prediction CV scores, data/code provenance, and `frozen_decisions.json`. Saved decision-training returns describe in-sample mixture performance; validation has been used for development and is not an untouched final test. Results exclude transaction costs and use daily Sharpe with risk-free rate zero. Turnover diagnostics account for drifted holdings.

The default run excludes August even when the source file contains it. Once the complete reserved August data are available, evaluate a saved model explicitly without recalibrating alpha or refitting/reselecting decision parameters:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_two_stage.py \
  --csv 100_Portfolios_10x10_Daily.csv \
  --frozen results/two_stage_lasso/frozen_decisions.json \
  --output results/two_stage_lasso_final_test
```

This path checks the frozen development-data fingerprint and all 20 reserved trading dates (August 3–28, 2026). Missing or changed history fails explicitly. See [the two-stage design](docs/superpowers/specs/2026-09-22-two-stage-lasso-decision-design.md) for the model and time boundaries.

The [September 22 experiment report](docs/experiments/two-stage-lasso.md) records the full comparison and independent audit. The selected three-year model gains little cumulative return over the old constrained 252-day LASSO while taking substantially more realized risk; the experiment does not establish an overall improvement.

## Direct asset allocation with linear programming

`run_direct_lp.py` uses the same causal 252-day and three-year LASSO forecasters, then directly solves all 98 daily asset weights with SciPy HiGHS. The decision stage has no trained mixture coefficients, regression, or epochs. Earlier runners remain available.

The LP maximizes predicted next-day return. Its fixed defaults require full investment, gross leverage at most 1.5, each absolute position at most 5%, historical daily wealth at least `1e-6`, and two historical risk limits relative to EW in the same three-year window:

- **MAD:** mean absolute deviation of portfolio daily returns around their historical mean.
- **Uncompounded absolute drawdown:** the largest decline in the cumulative **sum** of historical simple returns, including the initial zero cumulative return.

These are linear-programming risk measures. They are **not** standard deviation, a Sharpe constraint, or the compounded relative drawdown used to report performance. The runner separately evaluates actual daily Sharpe/volatility, compounded return, relative max drawdown, exposure and turnover. A historical risk constraint does not guarantee future realized risk. The secondary LP chooses holdings nearest to EW in L1 distance while preserving primary predicted return to a daily tolerance of `1e-10`; this does not train an EW mixture.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_direct_lp.py \
  --reference results/window_studies_3y/forecast_mixture_20260922 \
  --output results/window_studies_3y/forecast_lp_20260922
```

`--reference` is optional. When supplied, old forecasts/mixtures/direct-U/legacy baselines are compared only after checking data, dates, forecast settings, relevant code hashes, and reconstructing returns from saved weights. The default experiment has fixed risk settings and does not tune them on validation. Both historical-mean forecast controls and EW are evaluated. Alpha calibration remains before August 2024. The default run solves portfolios only for the January–June 2026 validation period (123 trading days); earlier history is still used for rolling LASSO training and risk constraints. Because the LP has no trained decision coefficients, it does not need a development-period portfolio backtest. Add `--include-development` only when that extra historical comparison is wanted.

To change LP limits explicitly, use `--position-limit`, `--mad-ratio`, and `--drawdown-ratio`, or pass `LPConstraints` through the API. Such changes are new experiments; infeasibility raises an error instead of substituting another portfolio. The output directory must be new or empty. Outputs include per-strategy forecasts, weights, returns, feasibility/turnover diagnostics, period summaries, provenance and frozen settings.

The reserved test requires all 20 August 3–28 trading dates and the unchanged development history, frozen source code and numerical dependency versions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_direct_lp.py \
  --frozen results/window_studies_3y/forecast_lp_20260922/frozen_decisions.json \
  --output results/direct_lp_final_test
```

The frozen path refits daily forecast coefficients with the saved alpha and solves new asset holdings, but does not reselect alpha or risk limits. See the [direct-LP report](docs/experiments/direct-lp.md) for definitions, validation results and limitations.
