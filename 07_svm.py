"""Experiment 7: compare linear and RBF support vector machines."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
for kernel in ("linear", "rbf"):
    model = SVC(kernel=kernel).fit(X_train, y_train)
    print(f"{kernel.upper()} SVM Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

