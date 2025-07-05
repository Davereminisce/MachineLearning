import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

from torchvision.models import resnet18, ResNet18_Weights

import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from torch.utils.tensorboard import SummaryWriter
start_time = time.time()
log_dir = f'logs/{int(start_time)}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
learning_rate = 0.0001
num_epochs = 20

in_channels = 3
num_class = 10


#Dataset
def data_tf(x):
    x = np.array(x, dtype='float32') / 255 
    x = (x - 0.5) / 0.5 
    x = cv2.resize(x, (224, 224)) 
    x = x.transpose((2, 0, 1)) 
    x = torch.from_numpy(x)
    return x
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_tf)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_tf)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#model构建————————————————————————————————————————————————————
resnet18_model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = resnet18_model
model.to(device)
#——————————————————————————————————————————————————————————————

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
