#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **作业 2: 音位分类 (分类)**

# 学习目标：
# * 数据预处理：从原始波形中提取MFCC特征
# * 分类：使用预提取的MFCC特征执行逐帧音位分类
# * 熟悉并提高pytorch训练技巧，熟悉pytorch模块
# 相关资料：
# * Slides地址: https://docs.google.com/presentation/d/1v6HkBWiJb8WNDcJ9_-2kwVstxUWml87b9CnA16Gdoio/edit?usp=sharing
# * Kaggle地址: https://www.kaggle.com/c/ml2022spring-hw2
# * 相关课程视频资源也可在B站获取  

# 前置知识：
# - 音位：
# 
# (phonetics 语音) 音位，音素（区分单词的最小语音单位，英语sip中的s和zip中的z是两个不同的音素）  
# 
# 例如：Machine Learning → M AH SH IH N L ER N IH NG M M M AH AH SH SH IH IH IH N N N N ... Machine
# 
# - MFCC：  
# 
# 在语音识别（Speech recognition）和话者识别（Speaker recognition）方面，最常用到的语音特征就是梅尔倒谱系数（Mel-scale Frequency Cepstral Coefficients, MFCC）。  
# 
# 根据人耳听觉机理的研究发现，人耳对不同频率的声波有不同的听觉敏感度。从200Hz到5000Hz的语音信号对语音的清晰度影响对大。两个响度不等的声音作用于人耳时，则响度较高的频率成分的存在会影响到对响度较低的频率成分的感受，使其变得不易察觉，这种现象称为掩蔽效应。由于频率较低的声音在内耳蜗基底膜上行波传递的距离（速度）大于频率较高的声音，故一般来说，低音容易掩蔽高音，而高音掩蔽低音较困难。在低频处的声音掩蔽的临界带宽较高频要小。所以，人们从低频到高频这一段频带内按临界带宽的大小由密到疏安排一组带通滤波器，对输入信号进行滤波。将每个带通滤波器输出的信号能量作为信号的基本特征，对此特征经过进一步处理后就可以作为语音的输入特征。由于这种特征不依赖于信号的性质，对输入信号不做任何的假设和限制，又利用了听觉模型的研究成果。因此，这种参数比基于声道模型的LPCC相比具有更好的鲁棒性，更符合人耳的听觉特性，而且当信噪比降低时仍然具有较好的识别性能。  
# 
# 在本次作业中，正常的一段音频素材可能包含大量的音位信息，而音位之间又可能存在重叠干扰的情况，因此我们将一段音频素材每隔10ms切取25ms，以此来尽可能保存完整的音位素材，取出后的素材称为一个frame，取出后的frame并不适合直接进入训练，因此我们要进行进一步的处理，通过MFCC，将它转化为一个39维度的特征，转换后为了更加精确的判断当前特征内的音位信息，我们往往采取其前后的特征来做辅助判断 也就是前向特征与后向特征各取5个，所以我们最后得到的是一个11*39维的一个向量。
# 
# ![](./pic/01.png) 
# 想要深入了解实现过程的可以查看下列链接：  
# [Prof. Hung-Yi Lee[2020Spring DLHLP] Speech Recognition](https://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/ASR%20(v12).pdf)  
# [ Prof. Lin-Shan Lee’s[Introduction to Digital Speech Processing]Chap.7](http://ocw.aca.ntu.edu.tw/ntu-ocw/ocw/cou/104S204)

# #  数据集下载
# 如果下列命令无法下载，可以到下列地址下载数据
# - Kaggle下载数据:  [Kaggle: ml2022spring-hw2](https://www.kaggle.com/competitions/ml2022spring-hw2)
# - 百度云下载数据: [云盘(提取码：05zc)](https://pan.baidu.com/s/198xn8Lk9MjvUsq866mZuuw)
# 
# 
# 下载完成后，你应该能够获取如下文件：
# - `libriphone/train_split.txt`
# - `libriphone/train_labels`
# - `libriphone/test_split.txt`
# - `libriphone/feat/train/*.pt`: training feature<br>
# - `libriphone/feat/test/*.pt`:  testing feature<br>  
# 
# pt文件可以使用torch.load方法导入
# 
# 
# <b>同学们下载完后直接解压到 HW02文件夹下面（将里面的文件最终放到HW02下）</b>

# In[ ]:


# 下载链接
get_ipython().system('wget -O libriphone.zip "https://github.com/xraychen/shiny-robot/releases/download/v1.0/libriphone.zip"')

# 下列数据获取方式需要依靠gdown

# 备用链接 0
# !pip install --upgrade gdown
# !gdown --id '1o6Ag-G3qItSmYhTheX6DYiuyNzWyHyTc' --output libriphone.zip

# 备用链接 1
# !pip install --upgrade gdown
# !gdown --id '1R1uQYi4QpX0tBfUWt2mbZcncdBsJkxeW' --output libriphone.zip

# 备用链接 2
# !wget -O libriphone.zip "https://www.dropbox.com/s/wqww8c5dbrl2ka9/libriphone.zip?dl=1"

# 备用链接 3
# !wget -O libriphone.zip "https://www.dropbox.com/s/p2ljbtb2bam13in/libriphone.zip?dl=1"

get_ipython().system('unzip -q libriphone.zip')
get_ipython().system('ls libriphone')


# In[2]:


# 输入如下指令查看GPU状态
get_ipython().system('nvidia-smi')


# ## 准备数据

# **Helper函数用于预处理来自每个话语的原始MFCC特征的训练数据**
# 
# 
# 一个音位可能跨越几个帧，并且取决于过去和将来的帧
# 
# 因此，我们连接相邻的音位进行训练以获得更高的准确性。**concat_fatte**函数连接过去和未来的k帧（总共2k+1＝n帧），我们预测中心帧。
# 
# 
# 可以随意修改数据预处理函数，但**不要删除任何帧**（如果您修改函数，请记住检查帧的数量是否与幻灯片中提到的相同）

# In[5]:


import os
import random
import pandas as pd
#导入pytorch
import torch
#导入进度条
from tqdm import tqdm
# 定义导入feature函数
def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)
# 将前后的特征联系在一起，如concat_n = 11 则前后都接5
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n 必须是奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

# 数据预处理函数
def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: 预先计算，不需要更改
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # 分割训练和验证数据
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X
# 返回的X代表数据的维度，如果不链接则为39 如果链接即为n*39 n为连接的特征总数,y为标签


# ## 定义数据集

# In[ ]:


import torch
#导入数据集
from torch.utils.data import Dataset
#导入数据加载工具Dataloader
from torch.utils.data import DataLoader
#定义数据集，一个数据集类应该包含初始化，_getitem__（获取一个元素）以及__len__（获取数据长度）方法
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# #  &#x2728; 神经网络模型
# <font color=darkred><b>***TODO***: 使用近似相同数量的参数实现2个模型，（A）一个更窄和更深（例如hidden_layers=6，hidden_dim＝1024）和（B）另一个更宽和更浅（例如 hidden_layers＝2、hidden_dim＝1700）。报告两种模型的训练/验证精度。</font></b>

# <font color=darkred><b>***TODO***:  添加dropout层，并报告dropout率分别等于（A）0.25/（B）0.5/（C）0.75的训练/验证准确性。</font></b>

# Dropout层在神经网络层当中是用来干什么的呢？它是一种可以用于减少神经网络过拟合的结构。
# ![](./pic/02.png)   
# 如上图我们定义的网络,一共有四个输入x_i，一个输出y。Dropout则是在每一个batch的训练当中随机减掉一些神经元，而作为编程者，我们可以设定每一层dropout（将神经元去除的的多少）的概率，在设定之后，就可以得到第一个batch进行训练的结果：  
# ![](./pic/03.png)   
# 从上图我们可以看到一些神经元之间断开了连接，因此它们被dropout了！dropout顾名思义就是被拿掉的意思，正因为我们在神经网络当中拿掉了一些神经元，所以才叫做dropout层。
# 在进行第一个batch的训练时，有以下步骤：
# * 设定每一个神经网络层进行dropout的概率
# * 根据相应的概率拿掉一部分的神经元，然后开始训练，更新没有被拿掉神经元以及权重的参数，将其保留
# * 参数全部更新之后，又重新根据相应的概率拿掉一部分神经元，然后开始训练，如果新用于训练的神经元已经在第一次当中训练过，那么我们继续更新它的参数。而第二次被剪掉的神经元，同时第一次已经更新过参数的，我们保留它的权重，不做修改，直到第n次batch进行dropout时没有将其删除。

# PS: 上面的两个TODO是可以改进其他部分来提高你的成绩的方法。  
# 如下的策略是助教给出的几个优化方式：  
# ● (1%) Simple baseline: 0.45797 (sample code)  
# ● (1%) Medium baseline: 0.69747 (concat n frames, add layers)  
# ● (1%) Strong baseline: 0.75028 (concat n, batchnorm, dropout, add layers)  
# ● (1%) Boss baseline: 0.82324 (sequence-labeling(using RNN))  
# 对于boss baseline，您可以参考RNN之前的课程记录

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
# 建立神经网络
class BasicBlock(nn.Module):# 继承 torch 的 Module
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)], # *[]将循环得到的解压
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# ## 超参数定义

# <font color=darkred><b>***TODO***:  可以考虑进一步优化超参数来提高准确率。</font></b>

# In[ ]:


# data prarameters
# 用于数据处理时的参数
concat_nframes = 1              # 要连接的帧数,n必须为奇数（总共2k+1=n帧）
train_ratio = 0.8               # 用于训练的数据比率，其余数据将用于验证
# training parameters
# 训练过程中的参数
seed = 0                        # 随机种子
batch_size = 512                # 批次数目
num_epoch = 5                   # 训练epoch数
learning_rate = 0.0001          # 学习率
model_path = './model.ckpt'     # 选择保存检查点的路径（即下文调用保存模型函数的保存位置）
# model parameters
# 模型参数
input_dim = 39 * concat_nframes # 模型的输入维度，不应更改该值，这个值由上面的拼接函数决定
hidden_layers = 1               # hidden_layer层的数量
hidden_dim = 256                # 隐藏维度


# ## 准备数据与模型

# In[3]:


# 引入gc模块进行垃圾回收
import gc

# 预处理数据
train_X, train_y = preprocess_data('train',phone_flie=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone',feat_dir=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone\feat', concat_nframes=concat_nframes,train_ratio=train_ratio)
val_X, val_y = preprocess_data('val',phone_flie=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone',feat_dir=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone\feat', concat_nframes=concat_nframes,train_ratio=train_ratio)

# 将数据导入
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# 删除原始数据以节省内存
del train_X, train_y, val_X, val_y
gc.collect()

# 利用dataloader加载数据
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# In[10]:


# 检查当前是否有可用的GPU 否则使用CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')


# In[11]:


import numpy as np

# 固定随机种子
def same_seeds(seed): # 固定随机种子（CPU）
    torch.manual_seed(seed) # 固定随机种子（GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True # 固定网络结构


# In[12]:


# 固定随机种子
same_seeds(seed)

# 创建模型、定义损失函数和优化器
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ## 训练模型

# In[13]:


best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # 训练部分
    model.train() # 设定模型到训练模式
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(features) 
        
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        
        _, train_pred = torch.max(outputs, 1) # 获得概率最高的类的索引
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    
    # 验证部分
    if len(val_set) > 0:
        model.eval() # 设定模型到评估模式
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                
                loss = criterion(outputs, labels) 
                
                _, val_pred = torch.max(outputs, 1) 
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # 获得概率最高的类的索引
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # 如果模型获得提升，在此阶段保存模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# 如果结束验证，则保存最后一个epoch得到的模型
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')


# In[14]:


del train_loader, val_loader
gc.collect()


# ## 测试
# 创建测试数据集，并从保存的检查点加载模型。

# In[15]:


# 载入数据
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# In[16]:


# 加载已经训练好的模型
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))


# In[17]:


test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # 获得概率最高的类的索引
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)


# 将预测结果写入CSV文件。

# In[18]:


with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))


# # 参考文献：  
# [一文入门dropout层](https://www.cnblogs.com/geeksongs/p/13446980.html)  
# 李宏毅机器学习2022在线课程

# # 贡献者  
# 潘笃驿(panduyi_azula@foxmail.com)
