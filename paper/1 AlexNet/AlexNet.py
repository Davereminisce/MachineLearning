import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm

# 数据预处理和加载

# CIFAR-10 数据集
# AlexNet原始输入是227x227，CIFAR-10是32x32，这里进行相应调整
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为224x224
    transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 下载并加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)


# # 定义自定义的CatDog数据集类
# class CatDogDataset(Dataset):
#     def __init__(self, txt_path, transform=None):
#         """
#         Args:
#             txt_path (string): 包含图片路径和标签的文本文件路径。
#             transform (callable, optional): 应用于样本的可选转换。
#         """
#         self.img_labels = []
#         # 读取文本文件，并解析图片路径和标签
#         with open(txt_path, 'r') as f:
#             for line in f:
#                 img_path, label = line.strip().split()
#                 self.img_labels.append((img_path, int(label)))  # 将标签转换为整数
#         self.transform = transform
#
#     def __len__(self):
#         # 返回数据集中样本的总数
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         # 根据索引加载图片和标签
#         img_path, label = self.img_labels[idx]
#         image = Image.open(img_path).convert('RGB')  # 确保图片为RGB三通道
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, torch.tensor(label)
#
# # 图像预处理和加载
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# # 使用自定义数据集加载训练和测试数据
# train_txt_path = os.path.join("data", "catVSdog", "train.txt")
# test_txt_path = os.path.join("data", "catVSdog", "test.txt")
#
# trainset = CatDogDataset(txt_path=train_txt_path, transform=transform)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
# testset = CatDogDataset(txt_path=test_txt_path, transform=transform)
# testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
#
# # 类别标签 (必须和你的txt文件中的标签对应)
# classes = ('cat', 'dog')

# 定义AlexNet模型架构
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一层：卷积层
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第二层：卷积层
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层：卷积层
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层：卷积层
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第五层：卷积层
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 自适应平均池化，将特征图大小调整为6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            # 展平特征
            nn.Dropout(),
            # 第六层：全连接层
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            # 第七层：全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 第八层：全连接层，输出类别数量
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 定义训练和评估函数
def train_model():
    # 检查是否有可用的CUDA设备，并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10
    model = AlexNet(num_classes=10).to(device)
    # # catdog
    # model = AlexNet(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    epochs = 10

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", unit="batch")
        for i, data in enumerate(pbar):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar_test = tqdm(testloader, desc=f"Epoch {epoch + 1}/{epochs} [Testing]", unit="batch")
            for data in pbar_test:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epochs} - Test Accuracy: {accuracy:.2f}%')

    # 保存 CIFAR-10 数据集 模型
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print('Finished Training and Model Saved.')

    # # 保存 catdog 数据集 模型
    # torch.save(model.state_dict(), 'alexnet_catdog.pth')
    # print('Finished Training and Model Saved.')


if __name__ == '__main__':
    train_model()
