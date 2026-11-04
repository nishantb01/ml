"""Experiment 10: compare a decision tree baseline, Random Forest, and AdaBoost."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
models = (DecisionTreeClassifier(random_state=42), RandomForestClassifier(n_estimators=100, random_state=42), AdaBoostClassifier(n_estimators=100, random_state=42))
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{model.__class__.__name__} accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nRandom Forest classification report:\n", classification_report(y_test, models[1].predict(X_test)))

