import os

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn


# model
"""
ResNet 模块化实现 
"""

# ------------------------------
# 基础残差块 BasicBlock (用于 ResNet-18/34)
# ------------------------------
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道不变
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # shortcut（捷径/残差连接）
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            # 将输入数据维度调整为输出维度
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ------------------------------
# 瓶颈残差块 Bottleneck (用于 ResNet-50/101/152)
# ------------------------------
class BottleneckBlock(nn.Module):
    expansion = 4  # 输出通道扩大 4 倍
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        width = out_planes
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ------------------------------
# Stage 构造函数
# ------------------------------
def make_stage(block, in_planes, out_planes, num_blocks, stride=1):
    """
    构造一个 stage (conv2_x / conv3_x / conv4_x / conv5_x)
    - block: BasicBlock 或 BottleneckBlock
    - in_planes: 输入通道
    - out_planes: stage 输出通道（Bottleneck 会再乘 expansion）
    - num_blocks: block 个数
    - stride: 第一个 block 的 stride (用于下采样)
    返回：nn.Sequential
    """
    layers = []
    layers.append(block(in_planes, out_planes, stride))
    in_planes = out_planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(in_planes, out_planes, stride=1))
    return nn.Sequential(*layers), in_planes


# ------------------------------
# 通用 ResNet
# ------------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_channels=3):
        super().__init__()
        self.in_planes = 64

        # 第一层卷积：7x7, stride=2, padding=3
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2..stage5
        self.stage2, self.in_planes = make_stage(block, self.in_planes, 64, layers[0], stride=1)
        self.stage3, self.in_planes = make_stage(block, self.in_planes, 128, layers[1], stride=2)
        self.stage4, self.in_planes = make_stage(block, self.in_planes, 256, layers[2], stride=2)
        self.stage5, self.in_planes = make_stage(block, self.in_planes, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 224 -> 112
        x = self.maxpool(x)                      # 112 -> 56
        x = self.stage2(x)                       # conv2_x
        x = self.stage3(x)                       # conv3_x
        x = self.stage4(x)                       # conv4_x
        x = self.stage5(x)                       # conv5_x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



# ------------------------------
# 工厂函数 (便于快速生成不同架构)
# ------------------------------
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes=10):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes=10):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes)


# ------------------------------
# 测试
# ------------------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 测试 ResNet34
#     model = ResNet34(num_classes=10).to(device)
#     print(model)
#
#     x = torch.randn(4, 3, 32, 32).to(device)
#     y = model(x)
#     print("ResNet34 输出:", y.shape)  # [4, 10]
#
#     # 测试 ResNet50
#     model50 = ResNet50(num_classes=10).to(device)
#     y50 = model50(x)
#     print("ResNet50 输出:", y50.shape)  # [4, 10]


# ------------------------------
# 数据预处理和加载
# ------------------------------
batch_size = 32

# CIFAR-10 数据集（32x32）
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

# 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ------------------------------
# 训练函数
# ------------------------------
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    return running_loss / total, 100. * correct / total

# ------------------------------
# 测试/验证函数
# ------------------------------
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / total
    acc = 100. * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")
    return avg_loss, acc

# ------------------------------
# 主函数
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 20
    learning_rate = 0.001

    # 创建保存模型的文件夹
    save_path = "./model"
    os.makedirs(save_path, exist_ok=True)
    best_acc = 0.0  # 记录最佳测试准确率

    # 模型
    model = ResNet34(num_classes=10).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练和测试
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        scheduler.step()

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_path, f"resnet34_best_{epochs}.pth"))
            print(f"Saved best model with acc: {best_acc:.2f}%; epoch:{epoch}")

    # 保存最终训练完成后的模型
    torch.save(model.state_dict(), os.path.join(save_path, f"resnet34_final_{epochs}.pth"))
    print("Saved final model after all epochs.")