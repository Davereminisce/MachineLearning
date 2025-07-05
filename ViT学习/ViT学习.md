# ViT学习

## 思考

### **ViT和CNN相比有什么区别**

#### **CNN**

CNN利用卷积层提取局部特征，用池化层来降低特征维度。

优点：

​	1.相比于ViT计算量少，效率高，占用内存小

​	2.相比于ViT在epoch小时更快出结果

缺点：

​	1.难以发现一个样本内部的关系

​	2.难以设置与样本相适应的CNN架构

#### **ViT**

ViT将图像划分为补丁，引入自注意力机制可以处理全局信息。

优点：

​	1.可以使模型对于每个样本都进行深度的剖析。能够发现每个样本内部的关系捕捉全局信息，进而使模型更加完善

​	2.通过改变参数便可调整模型，较为方便

缺点：

​	1.感觉容易产生过拟合情况

​	2.自注意力的计算复杂性高，内存和计算需求大

​	3.ViT训练慢，出结果慢

#### 模型准确率对比

最终训练模型的准确率还是与模型本身构建有关

对于ViT就是参数调整

对于CNN就是模型的选取，构建

## Transformer 模型学习

###### 第一版2024/10/19，新的版本在Task4.md中

Vision Transformer (ViT) 是 Transformer 模型针对计算机视觉任务的改编。所以我选择先进行Transformer 模型的学习

这里参照[csdn博客:一文读懂Transformer](https://blog.csdn.net/weixin_42475060/article/details/121101749?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159E0F39-AD8E-416D-AA92-7DD8F7B0E841%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&amp;request_id=159E0F39-AD8E-416D-AA92-7DD8F7B0E841&amp;biz_id=0&amp;utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~hot_rank-3-121101749-null-null.nonecase&amp;utm_term=Transformer%20%E6%A8%A1%E5%9E%8B%E5%9C%A8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD&amp;spm=1018.2226.3001.4450 )

Transformer是一种用于自然语言处理（NLP）和其他序列到序列（sequence-to-sequence）任务的深度学习模型架构

最重要的就是引入了自注意力机制，就是将输入的序列不同部分进行赋权重，从而达到构建输入的样本内部形成了关系

### 整体结构

![img](https://raw.githubusercontent.com/Davereminisce/image/4e662eaa018d603af6a1f349084301609ecba931/3319e3d6922a2e7f2499a3130d3b5925.png)

Encoder block对应编码器

Decoder block对应解码器

<img src="https://raw.githubusercontent.com/Davereminisce/image/b17a06c0902d09d0126539d07295f0cd08614bd6/20c5baff36eedc6100d9f107e4fe3c95.png" alt="img" style="zoom:50%;" />

这里有六个编码器叠加，6个解码器叠加（互相间没有共享参数）

编码器与解码器之间的简略结构

<img src="https://raw.githubusercontent.com/Davereminisce/image/721d773eb36530f2632236149590da91f894efb3/630deb7da181d99eb9dd7d70f6b4da98.png" alt="img" style="zoom:50%;" />

输入的句子先进入自注意层，将每个单词编码时也与其他单词相关联

解码器的注意力层也是关注整个输入的相关部分

### Encoder Block

#### 单头自注意力层

我认为注意力头就是为每一个样本都创建三个权重矩阵，注意力机制会将这些权重矩阵应用于输入特征，生成查询、键和值，进而计算注意力权重并整合信息，以此来建立样本内部的关系连接，捕捉样本内部的复杂关系。

<img src="https://raw.githubusercontent.com/Davereminisce/image/2b2defb7c73fa2e59dc0eeb8519c563eb496593b/49c88545fae57dbf255c8ab9fd279110.png" alt="img" style="zoom:50%;" />

##### 计算步骤

step1：先将输入向量进行词嵌入与三个权重矩阵进行相乘创建出查询向量Q，键向量K，值向量V

<img src="https://raw.githubusercontent.com/Davereminisce/image/bba3f7fada67549b4421bb4b703eb0aa2a65bf52/cc00beb97c344a486d07e3d9e8a58f06.png" alt="img" style="zoom:50%;" />

step2：计算自注意力层的输出。

<img src="https://i-blog.csdnimg.cn/blog_migrate/00994ceb6bf9e66db19611c496463364.png#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

整体的计算图

<img src="https://i-blog.csdnimg.cn/blog_migrate/e976d386a1aad85c2efb7fc965099c27.png#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

1.先将x与Wq、Wk、Wv进行矩阵相乘得出查询向量、键向量、值向量

2.查询向量*键向量得出分数

3.分数除以dk的平方根

4.对第三步的得分进行softmax归一化

4.softmax后的值*值向量求和后就是该单词在该句子中的注意力了

#### 多头自注意力层Multi——Head Attention

Multi是将这个关系建立得更加清晰，每个头都有自己的一组权重矩阵。

关键是分头

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a923f7bb907110650448d4b773bf0671.png#pic_center)

分头的实现：用多组Wq，Wk，Wv得到多组查询、键、值矩阵，然后每组分别计算得到一个Z矩阵。最后形成多个注意力头，然后再用一个矩阵将多头拼接，最后再与一个建立的附加权重矩阵相乘最后得到一个注意力头矩阵（相当于合成数据）

<img src="https://i-blog.csdnimg.cn/blog_migrate/fd6af04ca65df88d3f13f4aaf987b0f3.png#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

通览

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0b8dacfc201e24ef7dc0e690b41b998c.png#pic_center)

#### 位置编码使用

位置编码的存在是使同一个单词在不同位置的注意力分数不一样

这里选择词嵌入与位置编码进行相加，不是拼接（这里是防止维度增加）

<img src="https://raw.githubusercontent.com/Davereminisce/image/8c4e3b558e5910dc424af4a8e7e8ef031f54ce54/%7BA8129329-470C-42EB-B584-421725167A39%7D.png" alt="img" style="zoom:50%;" />

#### Add&Normalize

在经过多头注意力机制得到矩阵Z后，Z并没有直接传入全连接神经网络，而是需要经过一步Add&Normalize。

<img src="https://i-blog.csdnimg.cn/blog_migrate/29a24a78b70aa77ffd41b5ae2bfdc5e7.png#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

Add：在z的基础上加了一个残差块X，防止在深度神经网络的训练过程中发生退化的问题，退化的意思就是深度神经网络通过增加网络的层数，Loss逐渐减小，然后趋于稳定达到饱和，然后再继续增加网络层数，Loss反而增大。

Normalize：归一化，加快训练速度、提高训练的稳定性

这里选用的是LN可以每一维上进行归一化

#### Feed Forward（全连接层）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98f4c96bfa951f24b2d1b8c8686582bd.png)

在这里全连接层是一个两层的神经网络，先线性变换，然后ReLU非线性，再线性变换。

### Decoder Block

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/da0673543394c7623497447b1211054d.png#pic_center)

Decoder也是由6个decoder堆叠而成的。一个Decoder包含两个 Multi-Head Attention 层。第一个 Multi-Head Attention 层采用了 Masked 操作。第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。

#### Masked Multi-Head Attention

与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。

### 输出

首先经过一次线性变换，然后Softmax得到输出的概率分布，然后通过词典，输出概率最大的对应的单词作为我们的预测输出。

## ViT模型学习

学习参照[从头开始实现 Vision Transformer (ViT)](https://towardsdatascience.com/implementing-vision-transformer-vit-from-scratch-3e192c6155f0)

### ViT 架构概述

<img src="https://miro.medium.com/v2/resize:fit:1050/1*Q-mBZkDz7TUnVGw1KPwqOA.png" alt="img" style="zoom:80%;" />

将输入图像分割成小块，然后将其展平为向量序列。将这些向量由 Transformer 编码器处理，这使得模型能够通过自注意力机制学习块之间的相互作用。然后，Transformer 编码器的输出结果输入到分类层，该分类层输出输入图像的预测类别。

#### 将图像转换为嵌入（PatchEmbeddings）

将图像分割成不重叠的块网格

然后对其进行线性投影以获得每个块的固定大小的嵌入向量

```python
class  PatchEmbeddings (nn.Module): 
    """
    将图像转换成 patch，然后将其投影到向量空间中。
    """ 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.image_size = config[ "image_size" ] 
        self.patch_size = config[ "patch_size" ] 
        self.num_channels = config[ "num_channels" ] 
        self.hidden_size = config[ "hidden_size" ] 
        # 根据图像大小和 patch 大小计算 patch 数量
        self.num_patches = (self.image_size // self.patch_size) ** 2 
        # 创建一个投影层，将图像转换成 patch 
        # 该层将每个 patch 投影到大小为 hidden_size 的向量中
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size) 

    def  forward ( self, x): 
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x) 
        x = x.flatten( 2 ).transpose( 1 , 2 )
        return x
```

x.flatten(2)：张量 x从第二个维度开始展平。

如果 x 是一个形状为 (batch_size,height,width,channels)的四维张量，它会将 height 和width的维度合并，展平为(batch\_size, height *width, channels)的形状。

.transpose(1, 2)：表示交换第二个和第三个维度。

使用上面的例子，如果展平后的张量形状为 (batch_size,height *width,channels)，经过转置后，它将变为(batch_size,channels,height *width)。

##### 代码思考

参数变化情况

(batch_size, num_channels, image_size, image_size)

->(batch_size,hidden_size,num_patches平方根,num_patches平方根)

->(batch_size,hidden_size,num_patches)

->(batch_size, num_patches, hidden_size)

相当于将图片分成num_patches个补丁块，然后每个补丁块的特征维度是hidden_size

#### 将位置编码嵌入（Embeddings）

由于不同位置的块对最终预测的贡献可能不同，我们需要一种方法将块位置编码到序列中。

主要作用是将输入图像的补丁嵌入、分类标记和位置信息结合起来

```python
class  Embeddings (nn.Module): 
    """
    将 patch embeddings 与 class token 和 position embeddings 结合起来。
    """ 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.config = config 
        self.patch_embeddings = PatchEmbeddings(config) 
        # 创建可学习的 [CLS] token 
        # 与 BERT 类似，[CLS] token 添加到输入序列的开头
        # 并用于对整个序列进行分类
        self.cls_token = nn.Parameter(torch.randn( 1 , 1 , config[ "hidden_size" ])) 
        # 为 [CLS] token 和 patch embeddings 创建 position embeddings 
        # 将 [CLS] token 的序列长度加 1
        self.position_embeddings = nn.Parameter(torch.randn( 1 , self.patch_embeddings.num_patches + 1 , config[ "hidden_size" ])) 
        self.dropout = nn.Dropout(config[ "hidden_dropout_prob" ]) 

    def  forward ( self, x ): 
        x = self.patch_embeddings(x) 
        batch_size, _, _ = x.size() 
        # 将 [CLS] 标记扩展为批量大小
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
         cls_tokens = self.cls_token.expand(batch_size, - 1 , - 1 ) 
        # 将 [CLS] 标记连接到输入序列的开头
        # 这会导致序列长度为 (num_patches + 1)
         x = torch.cat((cls_tokens, x), dim= 1 ) 
        x = x + self.position_embeddings 
        x = self.dropout(x) 
        return x
```

##### 代码思考

###### cls_token

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
```

猜测是分类标记，为了创建个可以学习的（1，1，hidden_size）

这里是1只表示一个特定的标记

###### position_embeddings

```python
self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
```

应该是位置嵌入，形状为（1，num_patches+1，hidden_size）

可以将补丁和cls_token放在一起

这里num_patches + 1表示每个补丁和 [CLS] token 的位置都有对应的嵌入。

###### Dropout

```python
self.dropout = nn.Dropout(config["hidden_dropout_prob"])
```

用于在训练过程中减少过拟合。

###### cls_tokens

```
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
```

扩展分类标记，扩展到（batch_size，1，hidden_size）

###### torch.cat

```
x = torch.cat((cls_tokens, x), dim=1)
```

将 [CLS] tokens 连接到输入序列的开头（batch_size，num_patches+1，hidden_size）

这里cls_tokens中仅代表一个固定的标记不包含位置信息

###### 加入位置嵌入和 dropout

```
x = x + self.position_embeddings 
x = self.dropout(x)
```

将位置嵌入加到嵌入序列上

###### return

返回处理后的张量 x形状为 (batch_size,num_patches+1,hidden_size)。

#### 单个注意力头（AttentionHead）

将一系列嵌入作为输入，并计算每个嵌入的查询、键和值向量。然后使用查询和键向量来计算每个标记的注意力权重。然后使用注意力权重通过值向量的加权和来计算新的嵌入。（和Transformaer模型差不多）

```python
class  AttentionHead (nn.Module): 
    """
    单个注意力头。
    此模块用于 MultiHeadAttention 模块。
    """ 
    def  __init__ ( self, hidden_size,tention_head_size, dropout, bias= True ): 
        super ().__init__() 
        self.hidden_size = hidden_size 
        self.attention_head_size =tention_head_size 
        # 创建查询、键和值投影层
        self.query = nn.Linear(hidden_size,tention_head_size, bias=bias) 
        self.key = nn.Linear(hidden_size,tention_head_size, bias=bias) 
        self.value = nn.Linear(hidden_size,tention_head_size, bias=bias) 

        self.dropout = nn.Dropout(dropout) 
    
    def  forward ( self, x ): 
        # 将输入投影到查询、键和值中
        # 使用相同的输入来生成查询、键和值，
        # 因此通常称为自我注意。
        # (batch_size, serial_length, hidden_size) -> (batch_size, serial_length,tention_head_size)
        query = self.query(x) 
        key = self.key(x) 
        value = self.value(x) 
        # 计算注意力分数
        # softmax(Q*KT/sqrt(head_size))*
        attention_scores = torch.matmul(query, key.transpose(- 1 , - 2 )) 
        attention_scores =attention_scores / math.sqrt(self.attention_head_size) 
        attention_probs = nn. functional.softmax(attention_scores, dim=- 1 ) 
        attention_probs = self.dropout(attention_probs) 
        # 计算注意力输出
        tention_output = torch.matmul(attention_probs, value) 
        return (attention_output,attention_probs)
```

#### 多注意力头（MultiHeadAttention）

仍然和（Transformer模块差不多）

将所有注意力头的输出拼接再与一个建立的附加权重矩阵相乘最后得到一个注意力头矩阵

```python
class  MultiHeadAttention (nn.Module): 
    """
    多头注意力模块。
    该模块在 TransformerEncoder 模块中使用。
    """
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
        # 创建一个线性层，将注意力输出投影回隐藏大小
        # 在大多数情况下，all_head_size 和 hidden_size 是相同的
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size) 
        self.output_dropout = nn.Dropout(config[ "hidden_dropout_prob" ]) 

    def  forward ( self, x, output_attentions= False ): 
        # 计算每个注意力头的注意力输出
        attention_outputs = [head(x) for head in self.heads] 
        # 连接每个注意力头的注意力输出
        attention_output = torch.cat([attention_output for attention_output , _ in tention_outputs], dim=- 1 ) 
        # 将连接的注意力输出投影回隐藏大小
        attention_output = self.output_projection(attention_output) 
        attention_output = self.output_dropout(attention_output) 
        # 返回注意力输出和注意力概率（可选）
        if  not output_attentions: 
            return (attention_output, None )
        其他：
            attention_probs = torch.stack（[attention_probs for _，attention_probs intention_outputs ]，dim= 1）
            返回（attention_output，attention_probs）
```

#### Transformer 编码器

Transformer 编码器由一叠 Transformer 层堆叠而成，每个 Transformer 层由刚刚实现的多头注意力模块和前馈网络组成，为了更好地扩展模型和稳定训练，在 Transformer 层上添加了两层规范化层和跳跃连接。

##### MLP（多层感知机）

通常用于实现前馈神经网络。

```python
class  MLP (nn.Module): 
    """
    一个多层感知器模块。
    """ 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.dense_1 = nn.Linear(config[ "hidden_​​size" ], config[ "intermediate_size" ]) 
        self.activation = NewGELUActivation() 
        self.dense_2 = nn.Linear(config[ "intermediate_size" ], config[ "hidden_size" ]) 
        self.dropout = nn.Dropout(config[ "hidden_dropout_prob" ]) 

    def  forward ( self, x ): 
        x = self.dense_1(x) 
        x = self.activation(x) 
        x = self.dense_2(x) 
        x = self.dropout(x) 
        return x
```

##### Transformer块

```python
class  Block (nn.Module): 
    """
    单个 transformer 块。
    """ 
    def  __init__ ( self, config ): 
        super ().__init__() 
        self.attention = MultiHeadAttention(config) 
        self.layernorm_1 = nn.LayerNorm(config[ "hidden_​​size" ]) 
        self.mlp = MLP(config) 
        self.layernorm_2 = nn.LayerNorm(config[ "hidden_​​size" ]) 

    def  forward ( self, x, output_attentions= False ): 
        # 自我注意
        attention_output,attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions) 
        # 跳过连接
        x = x +attention_output 
        # 前馈网络
        mlp_output = self.mlp(self.layernorm_2(x)) 
        # 跳过连接
        x = x + mlp_output 
        # 返回 transformer 块的输出和注意概率（可选）
        如果 没有输出注意点：
            返回（x，None）
        否则：
      返回（x，attention_probs）
```

##### 堆叠多个 Transformer 层

```python
class  Encoder (nn.Module): 
    """
    变压器编码器模块。
    """ 
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
```

### ViT 用于图像分类

将使用 [CLS] 标记的嵌入传递到分类层。

以 [CLS] 嵌入为输入，输出每幅图像的 logit。

以下代码为用于图像分类的 ViT 模型：

```python
class  ViTForClassification (nn.Module): 
    """
    用于分类的 ViT 模型。
    """ 

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
        self.apply(self._init_weights)
        
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

    def  forward ( self, x, output_attentions= False ): 
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
```

### 设置模型配置

```python
config = { 
    “patch_size” ： 4 ，
    #表示将输入图像划分为多小块（patches）的大小
    “hidden_size” ： 48 ，
    #模型中每层的隐藏状态维度。即每个补丁块的特征维度数
    “num_hidden_layers” ： 4 ，
    #Transformer 编码器中的层数
    “num_attention_heads” ： 4 ，
    #在多头自注意力机制中使用的注意力头的数量。
    “intermediate_size” ： 4 * 48 ，
    #前馈神经网络中间层的大小，
    “hidden_dropout_prob” ： 0.0 ，
    #训练期间对隐藏层输出应用的丢弃率
    “attention_probs_dropout_prob” ： 0.0 ，
    #计算注意力权重时应用的丢弃率
    “initializer_range” ： 0.02 ，
    #初始化模型权重的范围
    “image_size” ： 32 ，
    #输入图像的大小
    “num_classes” ： 10 ，
    #分类的类别数量
    “num_channels” ： 3 ，
    #输入图像的通道数
    “qkv_bias” ： True ，
    #是否在查询、键和值的线性变换中使用偏置项
    "use_faster_attention": True,
    #是否使用更高效的注意力计算实现
}
```

#### 部分参数特定要求&解释

##### num_attention_heads

指的是在多头自注意力机制中使用的注意力头的数量。这个参数的设置通常与 hidden_size 相关。

<img src="https://raw.githubusercontent.com/Davereminisce/image/22a32f40403953d80a380c14b08dafebd354ce48/%7B6507EE1D-4D60-4035-B853-8A567EFA4080%7D.png" alt="img" style="zoom: 67%;" />

##### intermediate_size

指的是中间层（或隐藏层）的大小，可以按照自己需求设置为 hidden_size 的倍数

##### attention_probs_dropout_prob

用于控制 Transformer 模型中注意力机制的 dropout 比例。在训练期间，模型会随机“丢弃”一定比例的注意力权重，以防止过拟合。

设置这个值可以帮助提高模型的泛化能力，通常是在 0 到 1 之间的一个浮动值。例如，设置为 0.1 表示在训练时 10% 的注意力概率会被随机丢弃。

##### use_faster_attention

用于指定是否启用更高效的注意力计算实现。

启用此选项可能会提高训练和推理的性能，但也可能会引入一些数值稳定性或精度的问题。

### 设置model

```python
model = ViTForClassification(config).to(device)
```

# Task3basic.py

这里我尝试了自己构建ViT模型并自己设计参数

按照上面的代码直接复制粘贴，注意

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

# Task3ViT-B-16.py

[参考文章csdn](https://blog.csdn.net/Mathematic_Van/article/details/136346404?fromshare=blogdetail&sharetype=blogdetail&sharerId=136346404&sharerefer=PC&sharesource=Davereminisce&sharefrom=from_link)

这里借用预处理的ViT-B-16来进行构建，并在原本的Task2基础上进行复现

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
