import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

warnings.filterwarnings("ignore")
ROLL, SEED = "IMT2024040", 42
os.makedirs("predictions", exist_ok=True)
os.makedirs("results", exist_ok=True)

train = pd.read_csv(f"data/{ROLL}_train_var2.csv")
test = pd.read_csv(f"data/{ROLL}_test_var2.csv")
feats = [c for c in train.columns if c.lower().startswith("x")]
X, y, X_test = train[feats].values, train["y"].values, test[feats].values

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
ridge_alphas = np.logspace(-3, 3, 7)
lasso_alphas = [1e-3, 1e-2, 1e-1]

rows, best = [], {"cv_mse": np.inf}
for d in range(1, 21):
    for name, est, alphas in (
        ("ridge", Ridge(), ridge_alphas),
        ("lasso", Lasso(max_iter=5000, tol=1e-3), lasso_alphas),
    ):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scale2", StandardScaler()),
            ("reg", est),
        ])
        gs = GridSearchCV(pipe, {"reg__alpha": alphas}, cv=kf,
                          scoring=("neg_mean_squared_error", "r2"),
                          refit="neg_mean_squared_error", n_jobs=1)
        gs.fit(X, y)
        i = gs.best_index_
        r = {"degree": d, "model": name,
             "alpha": gs.best_params_["reg__alpha"],
             "cv_mse": -gs.cv_results_["mean_test_neg_mean_squared_error"][i],
             "cv_r2": gs.cv_results_["mean_test_r2"][i]}
        rows.append(r)
        if r["cv_mse"] < best["cv_mse"]:
            best = {**r, "estimator": gs.best_estimator_}
        print(f"var2 d={d:2d} {name:5s} alpha={r['alpha']:.1e} "
              f"MSE={r['cv_mse']:.4f} R2={r['cv_r2']:.4f}", flush=True)

res2 = pd.DataFrame(rows)
res2.to_csv("results/cv_results_var2.csv", index=False)

plt.figure(figsize=(6, 4))
for name, g in res2.groupby("model"):
    plt.plot(g["degree"], g["cv_mse"], "o-", label=name)
plt.yscale("log"); plt.xlabel("Polynomial degree"); plt.ylabel("CV MSE (log scale)")
plt.title("Problem 2: CV MSE vs degree"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/cv_curve_var2.png", dpi=150); plt.close()

print(f"==> var2 best: degree={best['degree']}, model={best['model']}, "
      f"alpha={best['alpha']:.1e}, MSE={best['cv_mse']:.4f}, R2={best['cv_r2']:.4f}",
      flush=True)

pd.DataFrame({"y": best["estimator"].predict(X_test)}).to_csv(
    f"predictions/{ROLL}-pred_var2.csv", index=False)
print("Saved predictions/IMT2024040-pred_var2.csv. Done.", flush=True)