# DBA5106 Portfolio Training Framework

一个以模块封装为重点的线性组合训练框架。数据、分割、模型、损失、正则化、样本权重、组合权重和评价相互独立，使用普通函数连接。

## 安装与运行

需要 Python 3.10+。在项目目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_example.py
python run_example.py --weighted --output results
python run_example.py --weighted --growth --output results/growth_252
python -m unittest discover -s tests -v
```

CSV 默认放在项目根目录，也可以用 `--csv /path/to/file.csv` 指定。`--start`、`--end` 和 `--window` 控制滚动评价区间和历史窗口。

默认使用 252 个交易日的训练窗口，比较 EW、MinVar、固定强度 LASSO/Ridge，以及每日重新选参的 LASSO_CV/Ridge_CV。`--weighted` 增加 RidgeWeighted、LASSO_CV_Weighted、Ridge_CV_Weighted，均同时使用 63 日半衰期的时间权重和 25% 的 EW 混合。原来的 `5106_assignement_01.py` 保留作为参考，新入口为 `run_example.py`。底层 `backtest`/`rolling_windows` 的默认窗口仍为 126；直接调用它们时请显式传 `window=252`。

## 模块与数据流

`CSV → 收益矩阵 → 历史窗口 → 回归问题 → 线性拟合 → 组合调整 → 样本外评价`

| 模块 | 主要接口 | 职责 |
|---|---|---|
| `data.py` | `load_returns`、`preprocess_returns` | 按表名读取、清洗、转换单位 |
| `splits.py` | `split_returns`、`rolling_windows` | 按时间分割与逐日滚动 |
| `model.py` | `RegressionProblem`、`fit_linear`、`LinearModel` | X/y 构造、固定线性结构、系数到权重映射 |
| `losses.py` | `weighted_mse`、`LogGrowthLoss` | 定义平方损失、方差与对数收益奖励，报告分项 |
| `regularization.py` | `Regularization` | 无正则、LASSO、Ridge |
| `constraints.py` | `PortfolioConstraints` | 总敞口、相对 EW 波动和财富下界约束 |
| `optimization.py` | `fit_growth` | 真正优化新损失的凸求解器 |
| `weights.py` | `uniform`、`exponential`、`shrink_to_equal` | 两类权重的独立变换 |
| `training.py` | `fit_portfolio`、`fit_portfolio_cv`、`fit_portfolio_growth` | 连接模型、损失、正则项和权重接口 |
| `evaluation.py` | `backtest`、`summarize` | 通用策略回测与日频评价 |

### 一次训练

```python
from functools import partial
from portfolio import Regularization, fit_portfolio, load_returns
from portfolio.weights import exponential, shrink_to_equal

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

- [固定 LASSO 的对数收益奖励实验（252 天窗口）](docs/experiments/log-growth-252.md)：完整 11 策略比较、约束对照、累计收益与总敞口图。
