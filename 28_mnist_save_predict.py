import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ========== 模型定义（保存和加载都要用同一份结构） ==========
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_and_save():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = SmallCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 只训练 1 个 epoch，加快速度（你已验证 2 epoch 能到 98%）
    model.train()
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

    # 保存
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("模型已保存为 mnist_cnn.pt")
    return test_set


def load_and_predict(test_set, n=5):
    model = SmallCNN()
    model.load_state_dict(torch.load("mnist_cnn.pt", weights_only=True))
    model.eval()

    print("\n单张预测示例：")
    with torch.no_grad():
        for i in range(n):
            image, label = test_set[i]  # image: (1, 28, 28)
            logits = model(image.unsqueeze(0))  # 增加 batch 维
            pred = logits.argmax(dim=1).item()
            print(f"样本{i}: 真实={label}, 预测={pred}")


if __name__ == "__main__":
    test_set = train_and_save()
    load_and_predict(test_set)
