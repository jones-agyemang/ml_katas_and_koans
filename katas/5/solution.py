"""
solution.py — Autograd is Not Magic
"""
import torch

def manual_linear_regression():
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y = 3 * X + 2 + 0.1 * torch.randn(100, 1)
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    learning_rate = 0.1

    for epoch in range(200):
        y_pred = X @ w + b
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            w.grad.zero_()
            b.grad.zero_()
        if epoch % 50 == 0:
            print(f"[Manual] epoch={epoch}, loss={loss.item():.6f}")

    print("\nManual parameters:")
    print("w:", w.item())
    print("b:", b.item())

def nn_linear_regression():
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y = 3 * X + 2 + 0.1 * torch.randn(100, 1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(200):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"[nn.Linear] epoch={epoch}, loss={loss.item():.6f}")

    print("\nnn.Linear parameters:")
    print("w:", model.weight.item())
    print("b:", model.bias.item())

def main():
    manual_linear_regression()
    print("\n" + "-" * 50 + "\n")
    nn_linear_regression()

if __name__ == "__main__":
    main()
