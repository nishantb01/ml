"""Experiment 1: missing values, categorical encoding, and scaling."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Self-contained Titanic-like sample so no external CSV is required.
df = pd.DataFrame({"Age": [22, 38, None, 35, 28], "Fare": [7.25, 71.28, 8.05, 53.10, None],
                   "Sex": ["male", "female", "female", "female", "male"],
                   "Embarked": ["S", "C", "S", None, "Q"]})
print("Original data:\n", df)
print("\nMissing values:\n", df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
for column in ("Sex", "Embarked"):
    df[column] = LabelEncoder().fit_transform(df[column])
df[["Age", "Fare"]] = StandardScaler().fit_transform(df[["Age", "Fare"]])
print("\nPreprocessed data:\n", df)

