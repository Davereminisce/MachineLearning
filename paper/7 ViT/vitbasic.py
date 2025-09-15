#这里我尝试了自己构建ViT但效果一般

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import time

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter#导入tensorboard
start_time = time.time()
log_dir = f'logs/{int(start_time)}'

#ViT model中的参数设置
config = {
    "patch_size":4,
    "hidden_size": 256,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,# 一般为hidden_size/64或者128
    "intermediate_size": 1024, # n * hidden_size
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

transform = transforms.Compose([ 
    transforms.ToTensor()          
])

#Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


#Model构建
#---------------------------------------------
#GELU激活函数声明
class NewGELUActivation(nn.Module):
    def __init__(self):
        super(NewGELUActivation, self).__init__()

    def forward(self, x):
        return x * torch.nn.functional.sigmoid(1.702 * x)
    
#将图像转换成 patch,然后将其投影到向量空间中。
class  PatchEmbeddings (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.image_size = config[ "image_size" ] 
        self.patch_size = config[ "patch_size" ] 
        self.num_channels = config[ "num_channels" ] 
        self.hidden_size = config[ "hidden_size" ] 

        self.num_patches = (self.image_size // self.patch_size) ** 2 

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size) 

    def  forward ( self, x): 
        x = self.projection(x) 
        x = x.flatten( 2 ).transpose( 1 , 2 )
        return x

#将 patch embeddings 与 class token 和 position embeddings 结合起来。
class  Embeddings (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.config = config 
        self.patch_embeddings = PatchEmbeddings(config) 

        self.cls_token = nn.Parameter(torch.randn( 1 , 1 , config[ "hidden_size" ])) 

        self.position_embeddings = nn.Parameter(torch.randn( 1 , self.patch_embeddings.num_patches + 1 , config[ "hidden_size" ])) 
        self.dropout = nn.Dropout(config[ "hidden_dropout_prob" ]) 

    def  forward ( self, x ): 
        x = self.patch_embeddings(x) 
        batch_size, _, _ = x.size() 

        cls_tokens = self.cls_token.expand(batch_size, - 1 , - 1 ) 

        x = torch.cat((cls_tokens, x), dim= 1 ) 
        x = x + self.position_embeddings 
        x = self.dropout(x) 
        return x

#单个注意力头。
class  AttentionHead (nn.Module): 
    def  __init__ ( self, hidden_size,tention_head_size, dropout, bias= True ): 
        super ().__init__() 
        self.hidden_size = hidden_size 
        self.attention_head_size =tention_head_size 

        self.query = nn.Linear(hidden_size,tention_head_size, bias=bias) 
        self.key = nn.Linear(hidden_size,tention_head_size, bias=bias) 
        self.value = nn.Linear(hidden_size,tention_head_size, bias=bias) 

        self.dropout = nn.Dropout(dropout) 
    
    def  forward ( self, x ): 
        query = self.query(x) 
        key = self.key(x) 
        value = self.value(x) 
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(- 1 , - 2 )) 
        attention_scores =attention_scores / math.sqrt(self.attention_head_size) 
        attention_probs = nn.functional.softmax(attention_scores, dim=- 1 ) 
        attention_probs = self.dropout(attention_probs) 
        # 计算注意力输出
        attention_output = torch.matmul(attention_probs, value) 
        return (attention_output,attention_probs)

#多头注意力模块。
class  MultiHeadAttention (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.hidden_size = config[ "hidden_size" ] 
        self.num_attention_heads = config[ "num_attention_heads" ] 
        # 注意力头大小是隐藏大小除以注意力头数量
        self.attention_head_size = self.hidden_size // self.num_attention_heads 
        self.all_head_size = self.num_attention_heads * self.attention_head_size 
        # 是否在查询、键和值投影层中使用偏差
        self.qkv_bias = config[ "qkv_bias" ] 
        # 创建注意力头列表
        self.heads = nn.ModuleList([]) 
        for _ in  range (self.num_attention_heads): 
            head = AttentionHead( 
                self.hidden_size, 
                self.attention_head_size, 
                config[ "attention_probs_dropout_prob" ], 
                self.qkv_bias 
            ) 
            self.heads.append(head) 

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size) 
        self.output_dropout = nn.Dropout(config[ "hidden_dropout_prob" ]) 

    def  forward ( self, x, output_attentions = False ): 
        # 计算每个注意力头的注意力输出
        attention_outputs = [head(x) for head in self.heads] 
        # 连接每个注意力头的注意力输出
        attention_output = torch.cat([attention_output for attention_output , _ in attention_outputs], dim=- 1 ) 
        # 将连接的注意力输出投影回隐藏大小
        attention_output = self.output_projection(attention_output) 
        attention_output = self.output_dropout(attention_output) 
        # 返回注意力输出
        if  not output_attentions:
            return (attention_output, None )

#MLP一个多层感知器模块。
class  MLP (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.dense_1 = nn.Linear(config[ "hidden_size" ], config[ "intermediate_size" ]) 
        self.activation = NewGELUActivation() 
        self.dense_2 = nn.Linear(config[ "intermediate_size" ], config[ "hidden_size" ]) 
        self.dropout = nn.Dropout(config[ "hidden_dropout_prob" ]) 

    def  forward ( self, x ): 
        x = self.dense_1(x) 
        x = self.activation(x) 
        x = self.dense_2(x) 
        x = self.dropout(x) 
        return x

#单个 transformer 块。
class  Block (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.attention = MultiHeadAttention(config) 
        self.layernorm_1 = nn.LayerNorm(config[ "hidden_size" ]) 
        self.mlp = MLP(config) 
        self.layernorm_2 = nn.LayerNorm(config[ "hidden_size" ]) 

    def  forward ( self, x, output_attentions= False ): 
        # 自我注意
        attention_output,attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions) 
        # 跳过连接
        x = x +attention_output 
        # 前馈网络
        mlp_output = self.mlp(self.layernorm_2(x)) 
        # 跳过连接
        x = x + mlp_output 

        return (x, attention_probs)

#变压器编码器模块。
class  Encoder (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        # 创建变压器块列表
        self.blocks = nn.ModuleList([]) 
        for _ in  range (config[ "num_hidden_layers" ]): 
            block = Block(config) 
            self.blocks.append(block) 

    def  forward ( self, x, output_attentions= False ): 
        # 计算每个块的变压器块的输出
        all_attentions = [] 
        for block in self.blocks: 
            x,attention_probs = block(x, output_attentions=output_attentions) 
            if output_attentions: 
                all_attentions.append(attention_probs) 
        # 返回编码器的输出和注意概率（可选）
        if  not output_attentions: 
            return (x, None ) 
        else : 
            return (x, all_attentions)

#ViT 用于图像分类
class  ViTForClassification (nn.Module): 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.config = config 
        self.image_size = config[ "image_size" ] 
        self.hidden_size = config[ "hidden_size" ] 
        self.num_classes = config[ "num_classes" ] 
        # 创建嵌入模块
        self.embedding = Embeddings(config) 
        # 创建转换器编码器模块
        self.encoder = Encoder(config) 
        # 创建线性层，将编码器的输出投影到类的数量
        self.classifier = nn.Linear(self.hidden_size, self.num_classes) 
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用正态分布初始化权重
                nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


    def  forward ( self, x, output_attentions= False): 
        # 计算嵌入输出
        embedding_output = self.embedding(x) 
        #计算编码器的输出
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions) 
        # 计算logits，将[CLS] token的输出作为分类的特征
        logits = self.classifier(encoder_output[:, 0 ]) 
        # 返回logits和注意力概率（可选）
        if  not output_attentions: 
            return (logits, None ) 
        else : 
            return (logits, all_attentions)

model = ViTForClassification(config).to(device)
#---------------------------------------------


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
        outputs, _= model(inputs)
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

#Test构建
def test(epoch):

    model.eval() 

    correct = 0
    total = 0
    with torch.no_grad():  
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _= model(images)
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

