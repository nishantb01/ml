"""Experiment 5: Ridge and Lasso regularization."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
for model in (Ridge(alpha=1.0), Lasso(alpha=0.01, max_iter=10000)):
    model.fit(X_train, y_train)
    print(f"{model.__class__.__name__} R2 Score: {r2_score(y_test, model.predict(X_test)):.4f}")

