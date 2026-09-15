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
python -m unittest discover -s tests -v
```

CSV 默认放在项目根目录，也可以用 `--csv /path/to/file.csv` 指定。`--start`、`--end` 和 `--window` 控制滚动评价区间和历史窗口。

首版使用固定的示例正则化强度，比较 EW、MinVar、LASSO、Ridge。`--weighted` 增加一个同时启用两类权重的 Ridge 示例。这些参数用于演示接口，未经过寻优。原来的 `5106_assignement_01.py` 保留作为参考，新入口为 `run_example.py`。

## 模块与数据流

`CSV → 收益矩阵 → 历史窗口 → 回归问题 → 线性拟合 → 组合调整 → 样本外评价`

| 模块 | 主要接口 | 职责 |
|---|---|---|
| `data.py` | `load_returns`、`preprocess_returns` | 按表名读取、清洗、转换单位 |
| `splits.py` | `split_returns`、`rolling_windows` | 按时间分割与逐日滚动 |
| `model.py` | `RegressionProblem`、`fit_linear`、`LinearModel` | X/y 构造、固定线性结构、系数到权重映射 |
| `losses.py` | `weighted_mse` | 定义归一化平方损失 |
| `regularization.py` | `Regularization` | 无正则、LASSO、Ridge |
| `weights.py` | `uniform`、`exponential`、`shrink_to_equal` | 两类权重的独立变换 |
| `training.py` | `fit_portfolio`、`FitResult` | 连接模型、损失、正则项和权重接口 |
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

`rolling_windows` 的起止日期应落在已有数据覆盖范围内；对超出末日的区间会报错，避免静默缩短评价期间。最终测试数据齐备并冻结方案后，才使用测试集的实际首末交易日回测。首版没有自动交叉验证或批量搜索；后续可用分割接口，在训练窗口内部选择参数再完整重拟合。

### 扩展方式

- **样本权重**：新增 `weight_fn(n) -> numpy.ndarray`，按历史行顺序返回 n 个有限非负权重，至少两个为正；传给 `sample_weights`。指数衰减按交易观察数计算半衰期。
- **组合调整**：新增 `adjust_fn(raw_weights) -> numpy.ndarray`，维持原资产顺序和权重和为 1；传给 `adjust_weights`。
- **拟合器**：新增 `fit_fn(X, y, *, regularization, sample_weight) -> LinearModel`，传给 `model_fit`。当前训练接口要求拟合器求解统一的平方损失目标。若新增其他损失，需一起扩展求解器和训练损失报告；不能只改损失显示函数。
- **完整策略**：实现 `strategy(history: DataFrame) -> Series`，返回以资产名为索引的权重，然后交给 `backtest`。策略返回的资产顺序可以不同，回测会按名称对齐。

## 数据与评价约定

- 只读取 `Average Equal Weighted Returns -- Daily` 表；默认按名称删除 `ME9 BM10`，不修改原始 CSV。
- 输入原始值 1 表示 1%，预处理后为 0.01。内部始终使用小数收益；不要重复预处理或再除以 100。
- `-99.99`、`-999` 转为 NaN，保留供检查；训练或回测用到它们会报错，不做隐式填补。
- 样本权重只作用于训练，默认评价不加权。Sharpe 使用日均收益除以样本标准差（`ddof=1`），不扣除无风险收益，不年化。零波动时 Sharpe 为 NaN。
- 累计收益使用复利；最大回撤从初始净值 1 计算，返回负数。回测假设每日按权重配置并持有一天，不计交易费用。
- 数据集当前截至 2026-07-31。删除指定列后，首个训练窗口至 2026-06-30 无缺失；七月 `BIG HiBM` 有 22 个缺失值，且尚无八月测试数据。默认验证区间有 123 个交易日，不能把它的结果称为最终测试成绩。

## 验证重点

测试涵盖 CSV 表段与单位、时间边界、解析加权 LASSO/Ridge、协方差 MinVar 等价性、样本权重缩放不变性、组合调整、未来信息隔离以及收益评价。真实数据检查使用文档的六资产 2026-01-02 权重和单日收益。
