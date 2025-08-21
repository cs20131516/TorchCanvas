# Auto-generated training skeleton for TorchCanvas export
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .model import ExportedModel

def demo_train(epochs=3, batch_size=8, C=9, T=256, num_classes=18, lr=1e-3, device='cpu'):
    model = ExportedModel().to(device)
    # dummy dataset
    X = torch.randn(128, C, T)
    y = torch.randint(0, num_classes, (128,))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model({"inp": xb})
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"[ep {ep}] loss={(total/len(ds)):.4f}")

if __name__ == "__main__":
    demo_train()
