
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
#新加入
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#新加入
from torch.utils.tensorboard import SummaryWriter

import time
log_dir = f'logs/{int(time.time())}'

train_file_path = r'boston-housing\train.csv'
test_file_path = r'boston-housing\test.csv'
num_epochs = 100
learning_rate = 0.0001
train_batch_size = 32

#TrainDataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
    
        data = data.dropna()
        
        self.len=data.shape[0]

        #转换为numpy型
        self.x_data = data.drop(columns=['medv', 'ID']).values
        self.y_data = data['medv'].values 

        for i in range(self.x_data.shape[1]):
            # 获取当前列
            col_data = self.x_data[:, i]
            # 计算四分位数（Q1, Q3）
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            # 定义异常值检测的上下界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 找到异常值的掩码
            col_outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            # 计算非异常值的均值
            non_outliers_mean = col_data[~col_outliers_mask].mean()
            # 替换异常值为均值
            col_data[col_outliers_mask] = non_outliers_mean
            # 更新列数据
            self.x_data[:, i] = col_data

        #特征标准化(需要numpy型)
        scaler = StandardScaler()
        self.x_data = scaler.fit_transform(self.x_data)

        #转化为torch tensors
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data, dtype=torch.float32)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset1 = DiabetesDataset(train_file_path)

train_loader = DataLoader(dataset = dataset1, 
                          batch_size = train_batch_size,
                          shuffle = True, #需要打乱
                          num_workers = 2)

#TestDataset
class DiabetesTestDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
      
        self.len=data.shape[0]

        #转换为numpy型
        self.Test_data = data.drop(columns=['ID']).values

        #特征标准化(需要numpy型)
        scaler = StandardScaler()
        self.Test_data = scaler.fit_transform(self.Test_data)

        #转化为torch tensors
        self.Test_data = torch.tensor(self.Test_data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.Test_data[index], 0

    def __len__(self):
        return self.len

dataset2 = DiabetesTestDataset(test_file_path)
test_loader = DataLoader(dataset = dataset2, 
                          batch_size = 1, 
                          shuffle = False,
                          num_workers = 2)


#Model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(13, 16)
        self.linear2 = torch.nn.Linear(16,32)
        self.linear3 = torch.nn.Linear(32,64)
        self.linear4 = torch.nn.Linear(64, 128)
        self.linear5 = torch.nn.Linear(128, 512)
        self.linear6 = torch.nn.Linear(512, 1024)
        self.linear7 = torch.nn.Linear(1024, 512)
        self.linear8 = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.relu(self.linear6(x))
        x = self.relu(self.linear7(x))
        y_pred = self.linear8(x)
        return y_pred
model = Model()


#调用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


#Loss and Optimizer
criterion = torch.nn.MSELoss()
criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training Cycle
#Train方法
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            x_data, y_data = data
            x_data, y_data = x_data.to(device), y_data.to(device)

            #前推
            y_pred = model(x_data)
            y_data = y_data.unsqueeze(1)
            loss = criterion(y_pred, y_data)

            #反馈
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        writer.add_scalar("Loss/epoch",running_loss,epoch+1)#新加入
        print(epoch+1, running_loss)


#Test方法
def test():
    print('--------------Test--------------')
    model.eval()

    with torch.no_grad(): 
        for idx, (Test_data, _) in enumerate(test_loader, 1):
            Test_data = Test_data.to(device)

            output = model(Test_data)
            price = output[0].item()

            writer.add_scalar("Predicted prices/ID",price, idx)#新加入

            print(idx, price)


if  __name__ == '__main__': 
    writer = SummaryWriter(log_dir)
    print(f'Logging to: {log_dir}')
    train()
    test()
    writer.close() #新加入

#启动tensorboard代码
#tensorboard --logdir= --port=6006