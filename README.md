# Polynomial Regression Assignment (Roll No: IMT2024040)

Two regression problems solved with polynomial feature expansion and cross-validated model selection.

- **Problem 1 (var1):** Steam turbine optimization. 6 inputs (x1-x6), target y, polynomial degrees 1-10.
- **Problem 2 (var2):** Subterranean thermal reservoir mapping. 3 spatial inputs (x1-x3), target y, polynomial degrees 1-20.

## Project structure

## Method
1. **var1:** Standardize features, expand to polynomial degree d (1-10), fit ordinary least squares, and score with 5-fold cross-validation (MSE and R2). The degree with the lowest CV MSE is selected.
2. **var2:** Scale features, expand to degree d (1-20), standardize the expanded features, and fit Ridge and Lasso with the regularization strength tuned by inner cross-validation. The (degree, model, alpha) combination with the lowest CV MSE is selected.
3. The chosen model is refit on the full training set and used to predict the test set.

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib
python src/train_and_predict.py
```
Predictions are written to `predictions/` as a single column with header `y`.

Note: `src/train_and_predict.py` runs Problem 1 (var1); `src/var2_only.py` runs Problem 2 (var2) with a lighter, single-process search.
