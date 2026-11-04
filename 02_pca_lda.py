"""Experiment 2: compare PCA and LDA on the Breast Cancer dataset."""
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
for name, reducer in (("PCA", PCA(n_components=2)), ("LDA", LinearDiscriminantAnalysis(n_components=1))):
    start = time.time()
    train_reduced = reducer.fit_transform(X_train, y_train) if name == "LDA" else reducer.fit_transform(X_train)
    test_reduced = reducer.transform(X_test)
    model = LogisticRegression(max_iter=2000).fit(train_reduced, y_train)
    print(f"{name} accuracy: {accuracy_score(y_test, model.predict(test_reduced)):.4f}")
    print(f"{name} elapsed: {time.time() - start:.4f} seconds")

