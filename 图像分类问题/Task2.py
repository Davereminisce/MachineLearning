##############################################
#Task2许多解释、想法都详细写在了对应的Task2.md中#
#Task2.py文件中的解释只是简化说明              #
##############################################


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


#检测GPU是否可用，并根据检测结果选择运行GPU还是CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 32             #批的大小：指在一次计算中同时进行（传入）的样本数量
learning_rate = 0.001       #学习率：优化器中的参数
num_epochs = 20             #训练循环的次数


#这里相当于Dataset

transform = transforms.Compose([    #transforms.Compose：用于将多个图像变换组合在一起的类。
    transforms.ToTensor()           #transforms.ToTensor()：将图像导入的数据类型转换为张量（Tensor）
])
# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


#这里是Model
#这里Net的思路写在了Task2.md中
#由最初的（Batch，3，32，32）转化为了（Batch，）
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 10, kernel_size = 3)
        self.conv2 = nn.Conv2d(10, 15, kernel_size = 3)
        self.conv3 = nn.Conv2d(15, 20, kernel_size = 5)
        self.conv4 = nn.Conv2d(20, 25, kernel_size = 5)
        self.pooling = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(400, 10)

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pooling(x)
        x = self.relu(self.conv4(x))
        x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# TODO:补全
model = Network().to(device)        #将模型移动到 GPU


#Loss and Optimizer构建
criterion = nn.CrossEntropyLoss()                   #选用交叉熵损失
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#选用对于图像分类常用的Adam

#这里是Train构建
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0              #累计记录损失
        correct = 0                     #记录训练数据是每次训练的正确数
        total = 0                       #
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)#存入GPU

            optimizer.zero_grad()       #为了清除梯度信息

            #forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() #累计记录损失

            _, predicted = torch.max(outputs.data, 1)   #找出最大概率维，就是对应的分类
            total += labels.size(0)                     #累计记录损失
            correct += (predicted == labels).sum().item()#预测分类与实际分类一样就表明预测正确

        accuracy = 100 * correct / total                 #计算正确率
        #输出显示
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():

    model.eval()        #进行模型评估、验证或推理

    #这后面与train基本解释一致
    correct = 0
    total = 0

    with torch.no_grad():   #指在模块中禁用梯度计算，相当于不backward，减少计算

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":      #这一行是为了让pytorch在Windows系统上运行（基础.md上有详细说明）
    train()
    test()
