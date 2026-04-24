"""
solution.py — Model ≠ Intelligence
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

def train_and_evaluate(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {title} ===")
    print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    X_balanced, y_balanced = make_classification(
        n_samples=2000, n_features=10, n_informative=5, n_redundant=2,
        weights=[0.5, 0.5], random_state=42
    )
    X_imbalanced, y_imbalanced = make_classification(
        n_samples=2000, n_features=10, n_informative=5, n_redundant=2,
        weights=[0.98, 0.02], flip_y=0.01, random_state=42
    )
    train_and_evaluate(X_balanced, y_balanced, "Balanced dataset")
    train_and_evaluate(X_imbalanced, y_imbalanced, "Imbalanced dataset")

if __name__ == "__main__":
    main()
