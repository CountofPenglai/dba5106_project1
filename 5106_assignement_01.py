import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

CSV_PATH = "/Users/julienros/Downloads/100_Portfolios_10x10_Daily.csv"
WINDOW = 126
ALPHAS = np.logspace(-8, 8, 21)

TEST_START = 20260801  # everything from here on is test-only: never read below this line

# The file stacks 4 tables (value-weighted, equal-weighted, avg firm size, #firms).
# The assignment requires the 2nd one: equal-weighted returns.
df = pd.read_csv(CSV_PATH, skiprows=26318, nrows=26296, index_col=0)
df = df.drop(columns=["ME9 BM10"])
df = df[df.index < TEST_START]  # drop test period so it can't leak into any step below

n_assets = df.shape[1]
w_ew = np.full(n_assets, 1 / n_assets)
N = np.vstack([np.eye(n_assets - 1), -np.ones(n_assets - 1)])

val2_start = df.index.get_loc(20260102)
val2_end = df.index.get_loc(20260630)

results = {"EW": [], "MinVar": [], "LASSO": [], "Ridge": []}

for t in range(val2_start, val2_end + 1):
    train = df.iloc[t - WINDOW : t].to_numpy()
    r_t = df.iloc[t].to_numpy()

    y = train @ w_ew
    X = train @ N

    beta_ols = LinearRegression().fit(X, y).coef_
    beta_lasso = LassoCV(alphas=ALPHAS, cv=5, max_iter=50000).fit(X, y).coef_
    beta_ridge = RidgeCV(alphas=ALPHAS, cv=5).fit(X, y).coef_

    w_minvar = w_ew - N @ beta_ols
    w_lasso = w_ew - N @ beta_lasso
    w_ridge = w_ew - N @ beta_ridge

    results["EW"].append(w_ew @ r_t)
    results["MinVar"].append(w_minvar @ r_t)
    results["LASSO"].append(w_lasso @ r_t)
    results["Ridge"].append(w_ridge @ r_t)

print(f"{'Model':<8} {'Mean':>8} {'Std':>8} {'Sharpe':>8}")
for name, rets in results.items():
    rets = np.array(rets)
    sharpe = rets.mean() / rets.std()
    print(f"{name:<8} {rets.mean():>8.4f} {rets.std():>8.4f} {sharpe:>8.4f}")
