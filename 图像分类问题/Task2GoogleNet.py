
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#新引入
import torch.nn.functional as F
import cv2
import numpy as np

import time

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from torch.utils.tensorboard import SummaryWriter
start_time = time.time()
log_dir = f'logs/{int(start_time)}'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
learning_rate = 0.0001
num_epochs = 20

in_channels = 3
num_class = 10


#Dataset（这里需要进行调整）
def data_tf(x):
    x = np.array(x, dtype='float32') / 255 # 归一化
    x = (x - 0.5) / 0.5  # 标准化
    x = cv2.resize(x, (224, 224)) #调整为224*224像素
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维
    x = torch.from_numpy(x) #转换为 Tensor 格式
    return x
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_tf)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_tf)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#model构建————————————————————————————————————————————————————
#conv-relu组合
class BasicConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) 
    
#Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels_1x1,
                 out_channels_1x1_3,  out_channels_3x3,
                 out_channels_1x1_5, out_channels_5x5,
                 out_channels_pool ):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, out_channels_1x1, 1)
 
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels_1x1_3, 1),
            BasicConv2d(out_channels_1x1_3, out_channels_3x3, 3, 1, 1)
        )
 
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channels, out_channels_1x1_5, 1),
            BasicConv2d(out_channels_1x1_5, out_channels_5x5, 5, 1, 2)
        )
 
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, out_channels_pool, 1)
        )
 
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        output = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(output, dim = 1)
    
#GoogLeNet构建
class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(GoogLeNet, self).__init__()
        #第一阶段
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1)
        )
        #第二阶段
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        )
        #第三阶段
        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )
        #第四阶段
        self.block4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1)
        )
        #第五阶段
        self.block5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, 1)
        )
        #全连接层
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_class),
        )
 
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
 
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc(x)

        return x
      
model = GoogLeNet(in_channels,num_class).to(device)
#————————————————————————————————————————————————————————————————————

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train构建
def train(epoch):
    model.train()
    running_loss = 0.0             #损失
    correct = 0                     
    total = 0                      
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()   

        #forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

        _, predicted = torch.max(outputs.data, 1)   
        total += labels.size(0)                    
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total                 #计算正确率

    writer.add_scalar('loss/epoch', running_loss, epoch+1)#传入loss到日志
    writer.add_scalar('accuracy/epoch', accuracy, epoch+1)#传入accuracy到日志       

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test(epoch):

    model.eval() 

    correct = 0
    total = 0

    with torch.no_grad():  

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    writer.add_scalar('Test_accuracy/epoch', accuracy, epoch+1)#传入loss到日志
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":

    writer = SummaryWriter(log_dir)
    print(f'Logging to: {log_dir}')

    for epoch in range(num_epochs):   
        train(epoch)
        test(epoch)
    writer.close()#训练结束后，关闭SummaryWriter

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')
