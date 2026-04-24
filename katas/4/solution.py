"""
solution.py — Leakage is Silent Death
"""
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def leaky_version(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=5, random_state=42).fit_transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def safe_pipeline_version(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=5, random_state=42)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    return accuracy_score(y_test, pipeline.predict(X_test))

def main():
    X, y = make_classification(
        n_samples=1500, n_features=30, n_informative=10, n_redundant=10,
        random_state=42
    )
    print("Leaky score:", round(leaky_version(X, y), 4))
    print("Safe pipeline score:", round(safe_pipeline_version(X, y), 4))
    print("\nThe safe score is the one you should trust.")

if __name__ == "__main__":
    main()
