
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from torch.utils.tensorboard import SummaryWriter#导入tensorboard
start_time = time.time()
log_dir = f'logs/{int(start_time)}'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
learning_rate = 0.001
num_epochs = 40



#Dataset
transform = transforms.Compose([ 
    transforms.ToTensor()          
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#ResNet的ResiduaBlock块设计(b,c,w,h)
class ResiduaBlock(nn.Module):
    def __init__(self, channels):
        super(ResiduaBlock, self).__init__() 
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(x+y)


#Model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size = 3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3)
        self.conv3 = nn.Conv2d(16, 20, kernel_size = 3)

        self.RBlock1 = ResiduaBlock(16)
        self.RBlock2 = ResiduaBlock(20)

        self.pooling = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(720, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.pooling(self.conv2(x)))
        x = self.RBlock1(x)
        x = self.relu(self.pooling(self.conv3(x)))
        x = self.RBlock2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet().to(device)


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
