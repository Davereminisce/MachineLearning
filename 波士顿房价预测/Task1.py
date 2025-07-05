import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

train_file_path = r'boston-housing\train.csv'
test_file_path = r'boston-housing\test.csv'
num_epochs = 10
learning_rate = 0.0001
train_batch_size = 32

#Dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        #data.describe()
    
        #对数据进行清洗，处理缺失值和异常值。
        data = data.dropna()
        
        self.len=data.shape[0]

        # 分离特征和目标变量
        #转换为numpy型
        self.x_data = data.drop(columns=['medv', 'ID']).values    # 转换为 numpy 数组
        self.y_data = data['medv'].values                   # 转换为 numpy 数组

        #使用替换异常值方法（需要选择处理numpy型）,使用四分位距法（IQR）来检测和替换异常值(这里直接套用)
        #for i in range(self.x_data.size(1)):# TypeError: 'int' object is not callable 代码中存在一个命名冲突
        for i in range(self.x_data.shape[1]):
            # 获取当前列
            col_data = self.x_data[:, i]
            # 计算四分位数（Q1, Q3）
            Q1 = np.percentile(col_data, 25)#之前写的torch.quantile是处理tensor型，需要改为np.percentile来处理numpy型
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
        #改正后的添加
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

        # 分离特征和目标变量
        #转换为numpy型
        self.Test_data = data.drop(columns=['ID']).values      # 转换为 numpy 数组

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


#Model(有两个梯度选择Model_x即每次下降x的梯度)
class Model_2(torch.nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.linear1 = torch.nn.Linear(13, 11)
        self.linear2 = torch.nn.Linear(11, 9)
        self.linear3 = torch.nn.Linear(9, 7)
        self.linear4 = torch.nn.Linear(7, 5)
        self.linear5 = torch.nn.Linear(5, 3)
        self.linear6 = torch.nn.Linear(3, 1)
        self.relu = torch.nn.ReLU()
        #self.relu =  torch.nn.LeakyReLU(negative_slope=0.01) 

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        y_pred = self.linear6(x)
        return y_pred
model = Model_2()   #选用2维度下降

class Model_4(torch.nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        self.linear7 = torch.nn.Linear(13, 9)
        self.linear8 = torch.nn.Linear(9, 5)
        self.linear9 = torch.nn.Linear(5, 1)
        self.relu = torch.nn.ReLU()
        #self.relu = torch.nn.LeakyReLU(negative_slope=0.01) 

    def forward(self, x):
        x = self.relu(self.linear7(x))
        x = self.relu(self.linear8(x))
        y_pred = self.linear9(x)
        return y_pred
#model = Model_4()  #禁用4维度下降

#调用GPU
device = torch.device("cuda")
model.to(device)


#Loss and Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion.to(device)

#Training Cycle
#Train方法
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            x_data, y_data = data
            x_data, y_data = x_data.to(device), y_data.to(device)#转至GPU

            #前推
            y_pred = model(x_data)
            y_data = y_data.unsqueeze(1)    # 改进：确保 y_data 维度与 y_pred 匹配
            loss = criterion(y_pred, y_data)

            #反馈
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(epoch+1, running_loss)


#Test方法
def test():
    print('--------------Test--------------')

    model.eval() #完成了Task2后进行的简化计算的优化

    with torch.no_grad():       #禁用梯度计算，相当于不用backward
        for idx, (Test_data, _) in enumerate(test_loader, 1):
            Test_data = Test_data.to(device)

            output = model(Test_data)
            print(idx, output[0].item())


if  __name__ == '__main__':     #让pytorch在Windows系统
    train()
    test()