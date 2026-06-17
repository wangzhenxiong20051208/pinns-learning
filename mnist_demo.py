# P1: 手写数字识别 (MNIST + PyTorch)
# 这是你学神经网络的第一课：认识手写数字0-9

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 下载数据（28x28像素的手写数字图片）
print("正在下载数据...")
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 2. 定义一个简单的神经网络
# 输入：784个像素点 → 隐藏层 → 输出：10个数字(0-9)的概率
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)   # 第一层：784 → 128
        self.fc2 = nn.Linear(128, 64)      # 第二层：128 → 64
        self.fc3 = nn.Linear(64, 10)       # 第三层：64 → 10
        self.relu = nn.ReLU()              # 激活函数

    def forward(self, x):
        x = x.view(-1, 28*28)              # 把28x28图片拉平成784个像素
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                    # 输出10个数字的得分
        return x

# 3. 创建网络、定义损失函数和优化器
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 4. 训练
print("开始训练...")
for epoch in range(5):                     # 训练5轮
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"第 {epoch+1} 轮  损失: {total_loss:.4f}")

# 5. 测试准确率
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"\n测试集准确率: {100 * correct / total:.2f}%")

# 6. 随机展示几个预测结果
images, labels = next(iter(test_loader))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    color = 'green' if predicted[i] == labels[i] else 'red'
    plt.title(f"真实:{labels[i].item()} 预测:{predicted[i].item()}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\n✅ 完成！绿色=预测正确，红色=预测错误")
