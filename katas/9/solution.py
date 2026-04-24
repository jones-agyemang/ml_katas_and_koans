"""
solution.py — The Curve Speaks
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    X, y = make_classification(
        n_samples=1200, n_features=20, n_informative=8, n_redundant=4,
        class_sep=0.8, flip_y=0.08, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    depths = list(range(1, 21))
    train_errors = []
    validation_errors = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_errors.append(1 - accuracy_score(y_train, model.predict(X_train)))
        validation_errors.append(1 - accuracy_score(y_val, model.predict(X_val)))

    for depth, train_error, val_error in zip(depths, train_errors, validation_errors):
        print(f"max_depth={depth:02d} | train_error={train_error:.4f} | validation_error={val_error:.4f}")

    plt.figure()
    plt.plot(depths, train_errors, marker="o", label="Training error")
    plt.plot(depths, validation_errors, marker="o", label="Validation error")
    plt.xlabel("Decision tree max_depth")
    plt.ylabel("Error")
    plt.title("Bias-Variance Diagnostic Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bias_variance_curve.png", dpi=150)
    print("\nSaved plot to bias_variance_curve.png")

if __name__ == "__main__":
    main()
