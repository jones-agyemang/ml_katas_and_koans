"""
solution.py — Overfit One Batch
"""
import torch
from torch import nn

class TinyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def main():
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 10
    num_classes = 3
    X_batch = torch.randn(batch_size, input_dim)
    y_batch = torch.randint(0, num_classes, size=(batch_size,))

    model = TinyClassifier(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()

    for step in range(500):
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y_batch).float().mean().item()
            print(f"step={step}, loss={loss.item():.6f}, accuracy={accuracy:.2f}")

    final_logits = model(X_batch)
    final_predictions = final_logits.argmax(dim=1)
    final_accuracy = (final_predictions == y_batch).float().mean().item()
    print("\nFinal batch accuracy:", round(final_accuracy, 4))
    print("Expected: close to 1.0")

if __name__ == "__main__":
    main()
