"""
solution.py — The Shape is Truth
"""
import numpy as np

def standardise_columns(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std

def covariance_matrix(X_standardised):
    n_samples = X_standardised.shape[0]
    return (X_standardised.T @ X_standardised) / (n_samples - 1)

def eigen_decomposition(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]

def main():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X_standardised = standardise_columns(X)
    cov = covariance_matrix(X_standardised)
    eigenvalues, eigenvectors = eigen_decomposition(cov)
    print("Original shape:", X.shape)
    print("Standardised column means:", X_standardised.mean(axis=0).round(6))
    print("Standardised column stds:", X_standardised.std(axis=0).round(6))
    print("\nCovariance matrix shape:", cov.shape)
    print(cov.round(4))
    print("\nEigenvalues:", eigenvalues.round(4))
    print("Eigenvectors shape:", eigenvectors.shape)

if __name__ == "__main__":
    main()
