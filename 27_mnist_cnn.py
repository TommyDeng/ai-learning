import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_set = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)


# 2. 小 CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 → 14x14
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


model = SmallCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 训练（为了速度只训 2 个 epoch）
device = torch.device("cpu")
model.to(device)

for epoch in range(4):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")

# 4. 测试准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {correct / total:.4f}")
