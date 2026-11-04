# Machine Learning Lab Practicals

Eleven standalone Python programs corresponding to the experiments in the ML lab manual. Each script includes runnable data: Scikit-learn's built-in datasets where suitable, and a small reproducible sample for feature engineering and polynomial regression. This avoids dependence on local CSV files and network downloads.

## Setup

```bash
python -m pip install -r requirements.txt
```

Run a practical from this folder, for example:

```bash
python 01_feature_engineering.py
```

Experiment 9 opens a Matplotlib plot. Experiment 11 uses four real grid-world actions; the manual's original sample selected an action but advanced to the next state independently of that action, so it was corrected here.

## Experiments

1. Feature engineering: missing-value imputation, label encoding, and standardization.
2. PCA and LDA dimensionality reduction with logistic regression.
3. Multiple linear regression and MSE, RMSE, and R².
4. Polynomial regression at degrees 2, 3, and 5.
5. Ridge and Lasso regularization.
6. Logistic regression with confusion matrix, precision, recall, and F1.
7. Linear and RBF SVM comparison.
8. K-Means inertia and silhouette scores.
9. DBSCAN clustering and noise detection.
10. Decision Tree, Random Forest, and AdaBoost comparison.
11. Q-Learning in a 4x4 grid world.

