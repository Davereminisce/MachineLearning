# LLM WaterMark

## Attention机制

就是自注意力机制嘛，在Task3.md（基于NLP的Transformer 模型学习）中已经学习做了相应笔记

## Transformer

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

2.查询向量*键向量得出分数（这里要注意乘的对象，别乱乘）

3.分数除以dk的平方根

4.对第三步的得分进行softmax归一化

4.softmax后的值*值向量求和后就是该单词在该句子中的注意力了

#### 多头自注意力层Multi——Head Attention

Multi是将这个关系建立得更加清晰，每个头都有自己的一组权重矩阵。

关键是分头

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a923f7bb907110650448d4b773bf0671.png#pic_center)

##### 分头的实现

用多组Wq，Wk，Wv得到多组查询、键、值矩阵，然后每组分别计算得到一个Z矩阵。最后形成多个注意力头，然后再用一个矩阵将多头拼接，最后再与一个建立的附加权重矩阵相乘最后得到一个注意力头矩阵（相当于合成数据）

<img src="https://i-blog.csdnimg.cn/blog_migrate/fd6af04ca65df88d3f13f4aaf987b0f3.png#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

##### 通览

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

这里选用的是LN是LN是在同一个样本中不同神经元之间进行归一化，可以每一维上进行归一化

<img src="https://i-blog.csdnimg.cn/blog_migrate/cc6ff426fbe027c7998efd23a8e0a833.png#pic_center" alt="在这里插入图片描述" style="zoom: 80%;" />

公式

```python
#第一个残差块
LayerNorm（X+MuitiHeadAttention（X））
#第二个残差块
LayerNorm（X+FeedForward（X））
```

#### Feed Forward（全连接层）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98f4c96bfa951f24b2d1b8c8686582bd.png)

在这里全连接层是一个两层的神经网络：

1.线性变换，模型学习输入数据的线性关系。

2.ReLU激活函数来引入非线性，让模型捕捉更复杂的特征。

3.再进行一次线性变换，将特征映射回所需的输出维度。

目的：为了让模型捕捉更复杂的特征

### Decoder Block

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/da0673543394c7623497447b1211054d.png#pic_center)

Decoder也是由6个decoder堆叠而成（Nx=6）。一个Decoder包含两个 Multi-Head Attention 层。第一个 Multi-Head Attention 层采用了 Masked 操作。第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。

#### Masked Multi-Head Attention

与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。这里有两种mask，padding mask、sequence mask（这里就不进行细致解释了）

#### Decoder Block的输入数据来源

Encoder中的Multi-Head Attention是基于Self-Attention，Decoder中的第二个Multi-Head Attention只基于Attention，它的输入Quer来自于Masked Multi-Head Attention的输出，Keys和Values来自于Encoder中最后一层的输出

### 输出

首先经过一次线性变换，然后Softmax得到输出的概率分布，然后通过词典，输出概率最大的对应的单词作为我们的预测输出。

## LLM 基本架构

大型语言模型（Large Language Model），是一种基于深度学习的人工智能模型，模型是为了理解和生成自然语言文本。如ChatGPT

### 特点

1.本身基于transformer架构

2.参数量巨大通常使用数十亿到数千亿的参数

3.需要通过大量的文本数据进行训练

### 主要类别框架

相关链接[scdnLLM大语言模型主要类别架构（二）](https://blog.csdn.net/pythonhy/article/details/139959836?fromshare=blogdetail&sharetype=blogdetail&sharerId=139959836&sharerefer=PC&sharesource=Davereminisce&sharefrom=from_link)

一般分为三种：自回归模型、自编码模型和序列到序列模型。

#### 自回归模型（AR）

如GPT，使用单词的上文来预测单词，适合自然语言生成任务。

从一系列time steps中学习，将上一步的结果作为回归模型的输入，预测下一个time step的值。通常用于生成式任务，在长文本的生成能力很强。

这里参照相关链接以AR代表模型GPT进行笔记记录

##### GPT模型架构

（同时与BERT（自编码模型类）作比较）

采用的是单向Transformer模型，给定一个句子[u1, u2, …, un], GPT在预测单词ui的时候只会利用[u1, u2, …, u(i-1)]的信息而BERT会同时利用上下文的信息[u1, u2, …, u(i-1), u(i+1), …, un]

 BERT采用Transformer的Encoder模块, 而GPT采用Transformer的Decoder模块. 并且GPT的Decoder Block和经典Transformer Decoder Block还有不同

![img](https://raw.githubusercontent.com/Davereminisce/image/4e662eaa018d603af6a1f349084301609ecba931/3319e3d6922a2e7f2499a3130d3b5925.png)

（经典的Transformer）

经典的Transformer Decoder Block包含3个子层, 分别是Masked Multi-Head Attention层, encoder-decoder attention层, 以及Feed Forward层. 但是在GPT中取消了第二个encoder-decoder attention子层, 只保留Masked Multi-Head Attention层, 和Feed Forward层。同时GPT的架构中采用了12个Decoder Block。（注意结合Transformer图像看）

##### GPT训练过程

第一阶段（预训练）: 无监督的预训练语言模型.

第二阶段（微调）: 有监督的下游任务fine-tunning.

##### AR模型总结

AR模型使用注意力机制，预测下一个token，适用于文本生成。但是AR模型只能用于前向或者后向建模，不能同时使用双向的上下文信息，不能完全捕捉token的内在联系。

#### 自编码模型 (AE)

以BERT为代表，利用上下文的双向信息进行预训练，擅长自然语言理解任务。

#### **列到序列模型 (Seq2Seq)** 

包含编码器和解码器，用于序列转换任务，如机器翻译。

# 水印技术

论文链接[A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226?context=cs.LG)

水印技术就是在数字内容中加入的一种可以识别的方法。

用于标识版权、确保内容的合法性和追踪使用情况。主要目的包括版权保护、防止滥用、质量控制和内容追踪。

水印可以是显式的（如可见的标志或文字）或隐式的（如在数据中嵌入特定的模式或信息）。

水印的设计需要具备鲁棒性，以抵御修改，同时保持内容的原始质量和自然度。

##### 鲁棒性

鲁棒性就是指模型在多种不同类型的干扰下的稳定性和适应能力。

鲁棒性能让水印技术不会被轻易破坏。

##  hard watermark

就是训练好的模型结合输入文本构建一个硬红色名单。模型会按照这个名单来没有红色名单的生成文本。

模型的哈希值是在训练时不断调整，最后在训练好的模型中变为了一个固定的哈希值。

在每次生成文本时通过这个固定的哈希值结合每个token的哈希值作为种子通过随机数生成器动态创建一个红色名单，最后融合形成一个最终的名单，再根据该名单生成文本。这样表明可以根据该模型和输入文本生成一个专属的名单。

该方法使名单具有动态效果。

### 生成硬水印名单的过程

首先训练好的模型会有一个固定的哈希值H

输入文本后，会根据文本产生第一个标记，用这标记哈希运算得到第一个哈希值。

再根据输入文本和前一个标记共同推断下一个标记，计算第二个哈希值，

以此类推。

在每次生成一个标记后使用固定的哈希值和当前标记的哈希值作为随机数生成器的种子，生成红色名单。（这里可看作的临时红色名单）

最终的红色名单是在所有通过标记生成的名单都生成完毕后，结合固定的哈希值对这些名单进行合并和调整，得出一个最终的红色名单。

该名单对应于该输入文本和模型。

### 检测水印的流程

总体思路是检测待测文本是否遵循一套特定的名单规则，有就代表是由该模型生成的。

首先使用已知的哈希函数和随机数生成器，对文本中的每个token基于其前一个token的值生成相应的红名单。最后生成一个由该待测文本得出的名单。

（这里假定是生成文本，那么该待测文本是由一套名单生成的输出文本，所以该文本推导出的名单与原名单是有一定关系的。）

将由输出文本推导出的名单进行文本遍历，统计绿色token数量|S|g，计算z=2（|S|g-T/2）/根号T，T是文本token总数，z越大代表该待测文本符合一个名单（推导出的名单），证明有名单存在，从而证明有水印存在。

### 移除水印的难度

通过一系列数学分析，需要更改很大比例的token才能降低z

### 硬红名单规则的缺点

我感觉缺点在于该方法建立的名单可能禁用很多token，从而使生成的文本多样性和流畅性变差。

越长的输入文本可能禁用的token越多。或者可能禁用了常接在某个token后的token。

## soft watermark

软水印实际上是在训练模型时对不同token的出现概率进行计算，然后在生成水印名单时引入参数。

对于高频率的token这个参数然其有更大几率分在绿色名单中，增强它们的出现概率。因为这些token在生成文本时通常会被选择。

对于使而低频率的token就随机分了。

### 生成软水印名单的过程

与硬水印名单生成过程差不多，只是在生成临时红色名单时，参数会影响绿色名单的token，增加它们的出现概率。

### 检测水印的流程

也是与硬水印检测方法差不多，只是要带入参数进行名单推导。

## 软水印与硬水印对比

模型训练：

软水印在训练模型过程中会计算每个令牌的参数和哈希值。

硬水印则只计算哈希值。

名单生成：

软水印在名单生成时引入参数根据令牌概率的动态调整，从而生成更加完善的名单。

硬水印则按照固定方法生成名单。

生成文本质量：

软水印对生成文本的影响较小。

硬水印可能会强制改变某些令牌的选择从而影响生成文本的质量。

检测：

软水印的检测与熵有关，高熵序列检测较为容易，低熵序列则需要更多令牌来进行检测。

硬水印则可以在任何情况下进行检测。

#### 熵的计算

这部分就是数学分析和不断测试。

感觉现在的水平下就直接用推荐的计算方法就行了

## Private Watermarking

这部分就是讲如何让水印更加安全（保护水印不被攻击）同时保持鲁棒性。

私有水印的意思就是设置了一系列安全保护措施的水印。

#### 安全保护措施：

使用随机密钥，保持密钥的秘密

通过安全API托管，增加了攻击者移除水印的难度。

使用一个伪随机函数。

防御暴力攻击

使用多个不同的密钥来降低通过频率分析发现水印的可能性。

## 攻击水印

可能存在的三种类型的攻击：文本插入、文本删除、文本替换

### 攻击类型

#### Paraphrasing Attacks（释义攻击）

通过手动或自动的方式对文本进行改写，以规避水印检测。

手动就是人为更改就行，自动就是再使用语言模型进行改写。

在套用另外一套名单，可能使其反推的名单复杂化

#### Discreet Alterations（细微修改）

进行小修改，比如改变单词拼写等。可能改变token读取

#### Tokenization Attacks（标记化攻击）

通过修改文本来影响后续单词的标记化。可能影响token的识别。

#### Homoglyph and Zero-Width Attacks（同形字和零宽攻击）

利用Unicode字符的同形字（形状相似但不同的字符）和零宽字符进行替换。可能影响标记token。

#### Generative Attacks（生成攻击）

利用大型语言模型的上下文学习能力，通过特定提示改变输出。从生成文本的阶段就改变水印的生成方法。

# 论文复现

这里直接使用论文作者发布在github的内容进行复现

[github](https://github.com/jwkirchenbauer/lm-watermarking)

根据README进行复现

#### 核心：

类：`WatermarkBase`、`WatermarkLogitsProcessor` 和 `WatermarkDetector`。

文件：

`watermark_processor.py`：提供最小实现。

`extended_watermark_processor.py`：提供更全面的实现（推荐）。

`demo_watermark.py`：实现 Gradio 演示界面。

模块：

`homoglyphs.py`和` normalizers.py`：实现 `WatermarkDetector` 使用的算法。

#### 水印超参数

##### 关键参数说明

- Delta：决定水印强度；适合的范围为 [0.5, 2.0]。
- 上下文宽度 (h)：上下文越长，检测越难，但对编辑的鲁棒性降低。
- 忽略重复 n-gram：检测时使用 `--ignore-repeated-ngrams=True` 以避免误导性 p 值。
- 伪随机函数 (PRF)：此选择仅在上下文宽度 `h>1` 时相关，并决定了上下文哈希的鲁棒性。`minhash` 被认为是最有效的。
- 自哈希：在不增加额外成本的情况下扩展上下文宽度，但可能会降低处理速度。
- Gamma：表示每个绿列表中词汇的比例。gamma=0.25更优，但这是一个小效果，合理的 `gamma` 值在 0.25 到 0.75 之间。
- 基础密钥：

参考默认设置

Gamma：0.25

Delta：2.0

上下文宽度 (h)： 4。（如果需要对编辑有更强的鲁棒性，可以降低 `h`）

伪随机函数（PRF）：默认是 `selfhash`，也可以使用 `minhash`。

在检测时，请始终使用 `--ignore-repeated-ngrams=True`。

#### 在代码中使用水印

生成水印文本

```python
from extended_watermark_processor import WatermarkLogitsProcessor

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash")
## 通过将播种方案设置为 `minhash` 可以关闭自哈希。

tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
output_tokens = model.generate(**tokenized_input,
                                logits_processor=LogitsProcessorList([watermark_processor]))
output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

```

检测水印文本

```python
from extended_watermark_processor import WatermarkDetector

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # 应与原始设置匹配
                                        seeding_scheme="selfhash", # 应与原始设置匹配
                                        device=model.device, # 必须与原始 rng 设备类型匹配
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

score_dict = watermark_detector.detect(output_text)

```

使用播种方案 `simple_1` 并在检测时将 `ignore_repeated_ngrams=False`。

#### 启动应用程序

这里有三种方法

选择简单使用

```
python app.py
```

`gradio`模块

这个库可以帮助你快速构建机器学习模型的用户界面。

`transformers`模块

这个库提供了强大的工具来处理各种自然语言处理任务。

`nltk`模块

NLTK 是一个非常强大的自然语言处理库，适用于多种文本处理任务。

这之后可以运行但是

![img](https://raw.githubusercontent.com/Davereminisce/image/ca2c870a37fdcd89b54fcf80fa418567b7826a7f/%7BE3A683A6-982D-4B86-A17F-C9DB498ED5D0%7D.png)

out of memory
