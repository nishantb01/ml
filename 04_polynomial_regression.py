"""Experiment 4: compare polynomial degrees on a reproducible salary-like dataset."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
X = np.arange(1, 31, dtype=float).reshape(-1, 1)
y = 25000 + 8500 * X.ravel() + 350 * X.ravel() ** 2 + rng.normal(0, 8000, len(X))
for degree in (2, 3, 5):
    features = PolynomialFeatures(degree=degree).fit_transform(X)
    model = LinearRegression().fit(features, y)
    print(f"Degree {degree} R2 Score: {r2_score(y, model.predict(features)):.4f}")

