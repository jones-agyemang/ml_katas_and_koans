"""
solution.py — Garbage In Still Wins
"""
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"\n{name}")
    print("Train accuracy:", round(train_accuracy, 4))
    print("Test accuracy:", round(test_accuracy, 4))

def main():
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    engineered = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    evaluate_model("Baseline logistic regression", baseline, X_train, X_test, y_train, y_test)
    evaluate_model("Logistic regression with polynomial features", engineered, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
