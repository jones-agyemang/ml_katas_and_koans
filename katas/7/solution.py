"""
solution.py — Abstraction Hides Truth

Requires:
    pip install tensorflow torch scikit-learn
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_data():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_with_keras(X_train, X_test, y_train, y_test):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\nKeras result:")
    print("loss:", round(loss, 4))
    print("accuracy:", round(accuracy, 4))

def train_with_pytorch(X_train, X_test, y_train, y_test):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(10):
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        predicted_labels = (test_predictions >= 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean().item()
        loss = loss_fn(test_predictions, y_test).item()

    print("\nPyTorch result:")
    print("loss:", round(loss, 4))
    print("accuracy:", round(accuracy, 4))

def main():
    X_train, X_test, y_train, y_test = build_data()
    train_with_keras(X_train, X_test, y_train, y_test)
    train_with_pytorch(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
