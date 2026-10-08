"""
Polynomial regression for two problems (Roll No: IMT2024040).
var1: degrees 1-10, plain OLS, 5-fold CV, pick by MSE (R2 reported).
var2: degrees 1-20, Ridge/Lasso with tuned alpha, pick by CV MSE.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # no display needed in WSL
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso

warnings.filterwarnings("ignore")

ROLL = "IMT2024040"
SEED = 42
DATA, PRED, RES = "data", "predictions", "results"
for d in (PRED, RES):
    os.makedirs(d, exist_ok=True)


# ---------- 1. LOADING DATA ----------
def load(var):
    """Return X_train, y_train, X_test using column names x1..xn and y."""
    train = pd.read_csv(f"{DATA}/{ROLL}_train_{var}.csv")
    test = pd.read_csv(f"{DATA}/{ROLL}_test_{var}.csv")
    feats = [c for c in train.columns if c.lower().startswith("x")]
    print(f"\n[{var}] train {train.shape}, test {test.shape}, features {feats}")
    print(train.describe().T[["mean", "std", "min", "max"]])
    print("Missing values in train:", int(train.isna().sum().sum()))
    return train[feats].values, train["y"].values, test[feats].values


def save_predictions(preds, var):
    """Single column, header 'y', no index (matches sample_submission.csv)."""
    path = f"{PRED}/{ROLL}-pred_{var}.csv"
    pd.DataFrame({"y": preds}).to_csv(path, index=False)
    print(f"Saved {path} ({len(preds)} rows)")


def plot_curve(df, var, title):
    """Plot CV MSE vs degree (log scale) so over/underfitting is visible."""
    plt.figure(figsize=(6, 4))
    plt.plot(df["degree"], df["cv_mse"], "o-", label="CV MSE")
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("CV MSE (log scale)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RES}/cv_curve_{var}.png", dpi=150)
    plt.close()


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------- 2. PROBLEM 1: degrees 1-10, OLS ----------
X, y, X_test = load("var1")
rows = []
for d in range(1, 11):
    # scale -> expand to degree d -> OLS
    model = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("ols", LinearRegression()),
    ])
    cv = cross_validate(model, X, y, cv=kf,
                        scoring=("neg_mean_squared_error", "r2"),
                        return_train_score=True)
    rows.append({
        "degree": d,
        "cv_mse": -cv["test_neg_mean_squared_error"].mean(),
        "cv_mse_std": cv["test_neg_mean_squared_error"].std(),
        "cv_r2": cv["test_r2"].mean(),
        "train_r2": cv["train_r2"].mean(),
    })
    print(f"var1 degree {d:2d}  MSE={rows[-1]['cv_mse']:.4f}  R2={rows[-1]['cv_r2']:.4f}")

res1 = pd.DataFrame(rows)
res1.to_csv(f"{RES}/cv_results_var1.csv", index=False)
plot_curve(res1, "var1", "Problem 1: CV MSE vs degree")
best_d = int(res1.loc[res1["cv_mse"].idxmin(), "degree"])
print(f"==> var1 best degree = {best_d}")
print(res1.round(4).to_markdown(index=False) if hasattr(res1, "to_markdown") else res1)

final1 = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=best_d, include_bias=False)),
    ("ols", LinearRegression()),
]).fit(X, y)
save_predictions(final1.predict(X_test), "var1")

print("\nVar1 done. Run src/var2_only.py for Problem 2.")
