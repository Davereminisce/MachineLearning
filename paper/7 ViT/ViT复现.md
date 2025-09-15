# ViT复现

# basic.py

这里我尝试了自己构建ViT模型并自己设计参数

1.调用math库

```python
import math
```

2.model改变

```python
model = ViTForClassification(config).to(device)
```

3.数据结构出现改变

根据ViTForClassification 类的forward方法，它会返回一个元组 (logits, all_attentions)，

而在train和test函数中需要outputs是一个张量

```python
#train函数中
outputs, _ = model(inputs)  # 只取 logits
#test函数中
outputs, _ = model(images)  # 只取 logits
```

### 训练结果

1.

batch_size = 32

lr = 0.01

num_epochs = 20

<img src="https://raw.githubusercontent.com/Davereminisce/image/fbff3b92f255a478edadd4718ddcdc5847978bb2/%7B27840506-0C81-4AA6-BC3F-829AB2417CA4%7D.png" alt="img" style="zoom:33%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/1edc4f8aa1f63205847bb290e7aeec01c400468b/%7B98FDCC2A-0914-4128-8057-B8F518C7CE02%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/d423132f3a54d540c38527eb260c1da1fc8312cd/%7B5A6D0E8D-6EB6-4444-89E0-FF0244D65474%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/3ec5156a879f1d3c4074735f1fcfac179fa0792b/%7B2C3F5BDF-E0E6-486E-8CF5-3E1C1B71479A%7D.png" alt="img" style="zoom:50%;" />

2.

```
batch_size = 64
learning_rate = 0.001
num_epochs = 100 
```

出现过拟合情况

<img src="https://raw.githubusercontent.com/Davereminisce/image/2de433a14aaeb1173905d9f1afae10dea75f6c9c/%7B2487CE23-5269-4819-B5E6-64FA7E3102D4%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/c3f36de258b2e78979dc84ada1c17160b478338c/%7B4DB945BC-FE61-4530-A47E-980BE71EEA44%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/d1e5fea224c29fe765010a6b3520d0a1e46aa19c/%7B4AB310F4-6676-4339-AA13-118545B9B0E3%7D.png" alt="img" style="zoom: 33%;" />

```python
#ViT model中的参数设置
config = {
    "patch_size": 4,
    "hidden_size": 256,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "intermediate_size": 1024,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# 全局变量
batch_size = 64
learning_rate = 1e-4
num_epochs = 50
```

<img src="https://raw.githubusercontent.com/Davereminisce/image/74a1366aaeafc2491c59f13935520438a6cacf0c/%7B65C3E098-EF15-4D46-81DA-D813A5A89E2B%7D.png" alt="img" style="zoom: 33%;" />

50次epoch用GPU训练都要3100s

<img src="https://raw.githubusercontent.com/Davereminisce/image/fbd04e623ae4e2aa5ed888d257ce6941bdb298b2/%7B979B5728-A24A-44D2-9AF3-AE5C815FFFC0%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/c8e9cc2131fe5c0aa38fda90724d2bddedfee8a5/%7BD98653F4-2F61-4892-9B1D-10185F00A177%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/a56684babdd30599938b92128bc98d8a971cc91f/%7BD9B16E08-6621-4148-ADDA-04DFCD4BBC3C%7D.png" alt="img" style="zoom:33%;" />

# ViT-B-16.py

```python
#导入
from torchvision.models import vit_b_16, ViT_B_16_Weights

#加载预训练的 ViT-B/16 权重
weights = ViT_B_16_Weights.DEFAULT
#使用加载的权重初始化 ViT-B/16 模型
model = vit_b_16(weights=weights)
#将模型的最后一层分类层替换为新的线性层
model.heads[0] = nn.Linear(model.heads[0].in_features, 10)
model.to(device)
```

### ViT-B-16模型学习

ViT-B-16模型相当于对于我Task3basic上进行参数调整

```python
config = {
    "patch_size": 16,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 224,
    "num_classes": 1000,
    "num_channels": 3, 
    "qkv_bias": True,
    "use_faster_attention": True, 
}

batch_size = 32                 # 批量大小
learning_rate = 1e-4            # 学习率
num_epochs = 100                # 训练周期
```

注意这个需要将数据集更改为224*224

可以参考Task2的数据处理

```python
def data_tf(x):
    x = np.array(x, dtype='float16') / 255 # 改为16位减少计算
    x = (x - 0.5) / 0.5  # 标准化
    x = cv2.resize(x, (224, 224)) #调整为224*224像素
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维
    x = torch.from_numpy(x) #转换为 Tensor 格式
    return x
    
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_tf)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_tf)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
```

### 训练结果

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 64

learning_rate = 0.001

num_epochs = 10

（这里计算量过大，训练时GPU占用率已经达100%了）

<img src="https://raw.githubusercontent.com/Davereminisce/image/bdce16e6db975363c1a76aabf411b4670753ec0a/%7B5C011AAC-6A9C-4256-8465-4E1E19E76D32%7D.png" alt="img" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Davereminisce/image/8a488b325194c0b79d708a664cc76f159cd4685b/%7B1097D2A5-FED2-4A5C-B414-638D2C306B07%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/4425984e3963625bc775e829ddb2d00eff172830/%7BD6EC4C2A-B99E-42EA-BEAD-60238192F86A%7D.png" alt="img" style="zoom:33%;" /><img src="https://raw.githubusercontent.com/Davereminisce/image/027ad0932be8d9a62ace805859e0d0e332a641a3/%7B5DA88595-EF28-4AB3-8772-CBD8CD02C635%7D.png" alt="img" style="zoom:33%;" />

由于时间原因我只记录了7次，但是不难推断，这个模型最终准确率应该十分高。
