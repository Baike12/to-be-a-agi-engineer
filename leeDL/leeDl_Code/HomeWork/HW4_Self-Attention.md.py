#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **HW4 音频分类**
# - 给定音频区分出说话的人
# - 主要目标: 学会使用transformer
# - Baselines:
#   - Easy: 知道怎么使用transformer, 输出简单的可以运行的脚本.
#   - Medium: 知道transformer如何调参.
#   - <font color=darkred><b>Strong: 改变transformer的结构，使用一种transformer变体—— [conformer](https://arxiv.org/abs/2005.08100)  </font></b>
#   - <font color=darkred><b>Boss: 使用 [Self-Attention Pooling](https://arxiv.org/pdf/2008.01077v1.pdf) & [Additive Margin Softmax](https://arxiv.org/pdf/1801.05599.pdf)进一步提升模型表现. </font></b>
# 
# 
# - 其他链接：
#   - Kaggle: [link](https://www.kaggle.com/t/ac77388c90204a4c8daebeddd40ff916)
#   - Slide: [link](https://docs.google.com/presentation/d/1HLAj7UUIjZOycDe7DaVLSwJfXVd3bXPOyzSb6Zk3hYU/edit?usp=sharing)
#   - Data: [link](https://drive.google.com/drive/folders/1vI1kuLB-q1VilIftiwnPOCAeOOFfBZge?usp=sharing)

# # **加载包**

# In[14]:


get_ipython().system('pip install torchviz')


# In[15]:


import pandas as pd 
import numpy as np
import random
from pathlib import Path

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, random_split, DataLoader
from torch import functional  as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
import math

import os
import sys
import json
from tqdm import tqdm
# 绘制评估曲线
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchviz import make_dot

import warnings 
from rich.console import Console
warnings.filterwarnings('ignore')
cs = Console()


# # **下载数据**
# ```python
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partaa
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partab
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partac
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partad
# !cat Dataset.tar.gz.part* > Dataset.tar.gz
# # unzip the file
# !tar zxvf Dataset.tar.gz
# ```
# 如果https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/ 下载不了可以用以下途径下载数据
# - [Kaggle Data: ml2022spring-hw4](https://www.kaggle.com/competitions/ml2022spring-hw4/data)

# # **一些重要的函数**
# - all_seed 设置随机种子

# In[16]:


def model_plot(model_class, input_sample):
    clf = model_class()
    y = clf(input_sample) 
    clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [('x', input_sample)]))
    return clf_view


def all_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')


# In[17]:


all_seed(87)


# # **数据集**
# - 原始数据集 [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
# - The [license](https://creativecommons.org/licenses/by/4.0/) and [complete version](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) of Voxceleb2.
# - 我们从Voxceleb2数据集中随机抽取600个演讲者 
# - 将数据原始波形转换为mel谱图
# 
# - 文件夹的结构如下:
#   - data directory   
#   |---- metadata.json    
#   |---- testdata.json     
#   |---- mapping.json     
#   |---- uttr-{random string}.pt   
# 
# - metadata.json中的信息
#   - "n_mels": 40， mel图谱的维度.
#   - "speakers": 字典. 
#     - Key: speaker ids.
#     - value: "feature_path"-特征文件 and "mel_len"-特征的长度
# 
# 
# 为了更加高效, 我们在训练的时候将mel图谱分割成一定的长度。

# In[18]:


# 查看数据
data_dir = '../input/ml2022spring-hw4/Dataset'

# mapping.json
map_ = Path(data_dir) / 'mapping.json'
map_js = json.load(map_.open())
cs.print('mapping.json | keys = ', map_js.keys())
# metadata.json 
matedata_ = Path(data_dir) / 'metadata.json'
matedata_js = json.load(matedata_.open())
cs.print('metadata.json | keys = ', matedata_js.keys())
cs.print("matedata_js['n_mels']=", matedata_js['n_mels'])
cs.print(
    "matedata_js['speakers']['id00559'][:5]=", 
    matedata_js['speakers']['id00559'][:5]
)

mel = torch.load(os.path.join(data_dir, 'uttr-2918eae600684146903d49f02275cb94.pt'))
cs.print(f'(mel_len, n_mels)={mel.shape}') 
mel


# In[19]:


class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        super(myDataset, self).__init__()
        self.data_dir = data_dir
        self.segment_len = segment_len
        # 加载演讲者和id编码的映射.
        mapping_path = Path(data_dir) / 'mapping.json'
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']
        
        # 加载训练数据的源数据(特征文件， 演讲者)
        metadata_path = Path(data_dir) / 'metadata.json'
        metadata = json.load(metadata_path.open())['speakers']
        
        # 获取总演讲者数
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker, utt in metadata.items():
            for utt_i in utt:
                self.data.append([utt_i['feature_path'], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # 载入经过预处理的mel图谱特征(mel-spectrogram)
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        # 分割 mel-specrogram
        if len(mel) > self.segment_len:
            # 开始的位置为随机
            start = random.randint(0, len(mel) - self.segment_len)
            # 切分语音
            mel = torch.FloatTensor(mel[start: start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # 将speaker 转成long格式便于后续计算loss
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
    
    def get_speaker_number(self):
        return self.speaker_num


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        super(InferenceDataset, self).__init__()
        test_path = Path(data_dir) / 'testdata.json'
        metadata = json.load(test_path.open())
        self.data_dir = data_dir 
        self.data = metadata['utterances'] 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utt = self.data[index]
        feat_path = utt['feature_path']
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel
    
    
def inference_collate_batch(batch):
    feat_path, mels = zip(*batch)
    return feat_path, torch.stack(mels)


# ##  &#x2728;  **Transformer模型**
# <font color=darkred><b>***TODO***: encode改用Conformer</font></b>  
# <font color=darkred><b>***TODO***: 增加Self-Attention Pooling Layer</font></b>  
# 
# - 可以参考[https://github.com/sooftware/conformer](https://github.com/sooftware/conformerhttps://github.com/sooftware/conformer)
# - self-attetion & multi-self-attention & transformer block可以看李老师的视频
#     - [B站视频 第五讲 Transformer-2](https://www.bilibili.com/video/BV1m3411p7wD?p=33&vd_source=f209dda877a0d7be7d5309f93b340d6f)

# In[20]:


class Classifier(nn.Module):
    def __init__(self, input_dim=40, d_model=80, n_spks=600, dropout=0.1):
        super(Classifier, self).__init__()
        self.pre_net = nn.Linear(input_dim, d_model)
        # TODO:
        #   尝试改变Transformer， 改成Conformer.
        #   https://arxiv.org/abs/2005.08100 
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # self_attn [Q, K, V] shape=(d_model*3, d_model)
            dim_feedforward=256,
            nhead=2, 
            batch_first=True,
            activation='gelu'
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_spks)
        )
    
    def forward(self, mels):
        # out: (batch_size, length, d_model)
        out = self.pre_net(mels)
        out = self.encoder_layer(out)
        # mean pooling
        stats = out.mean(dim=1)
        return self.pred_layer(stats)


# In[21]:


# x = torch.randn(1, 100, 40).requires_grad_(True)
# model_plot(Classifier, x)


# ## &#x1F526;**学习率设置**
# - 对于transformer结构, 学习率的设计和CNN有所不同
# - 一些相关工作表明在训练前期逐步增加学习率（Warm up）有利于模型训练transformer.
# - 按照`plot_lr`设计一个Warm up的学习变化架构
#   - 设置学习率在 0到优化器设置的学习率的区间
#   - 在初期（Warmup period）学习率从零增长到0 to 初始学习率

# In[37]:


def plot_lr():
    num_warmup_steps=1000
    num_training_steps=70000
    lr = 0.01
    res_list = []
    for current_step in range(70000):
        if current_step < num_warmup_steps:
            res = float(current_step) / float(max(1, num_warmup_steps))
            res_list.append(res * lr)
            continue
        progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
        res = 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress))
        res_list.append(res * lr)

    plt.plot(res_list)
    plt.title(f'Trend of Learning Rate\nnum_warmup_steps={num_warmup_steps}\nnum_training_steps={num_training_steps}')
    plt.show()

plot_lr()


# In[22]:


def get_cosine_schedule_with_warmup(
    opt: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    创建一个学习率变化策略,
    学习率跟随cosine值变化,
    在warm up时间段内变化区间在:
        0 -> 优化器设置的学习率 .
    Args:
        opt (Optimizer): 优化器类
        num_warmup_steps (int): 多少步增加一下lr
        num_training_steps (int): 总训练步骤
        num_cycles (float, optional): 变化周期. 默认为 0.5.
        last_epoch (int, optional): _description_. Defaults to -1.
    """
    def lr_lambda(current_step):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 衰减
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(opt, lr_lambda, last_epoch)


# #   &#x2728; **训练部分**
# 这部分和HW01 & HW03基本相同

# In[29]:


def trainer(train_loader, valid_loader, model, config, device, rest_net_flag=False):
    # 对于分类任务, 我们常用cross-entropy评估模型表现.
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 
    if config['scheduler_flag']:
        scheduler = get_cosine_schedule_with_warmup(optimizer, config['warmup_steps'], len(train_loader) * config['n_epochs'])
    # 模型存储位置
    save_path =  config['save_path']

    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_accs = []
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()             
            x, y = x.to(device), y.to(device)  
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if config['scheduler_flag']:
                scheduler.step()
            step += 1
            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            l_ = loss.detach().item()
            loss_record.append(l_)
            train_accs.append(acc.detach().item())
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})
        
        
        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('ACC/train', mean_train_acc, step)
        
        model.eval() # 设置模型为评估模式
        loss_record = []
        test_accs = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()

            loss_record.append(loss.item())
            test_accs.append(acc.detach().item())
            
        mean_valid_acc = sum(test_accs) / len(test_accs)
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f},acc: {mean_train_acc:.4f} Valid loss: {mean_valid_loss:.4f},acc: {mean_valid_acc:.4f} ')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('ACC/valid', mean_valid_acc, step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), save_path) # 保存最优模型
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


# # **参数设置**

# In[30]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 87,
    'dataset_dir': "../input/ml2022spring-hw4/Dataset",
    'n_epochs': 35,      
    'batch_size': 64, 
    
    'scheduler_flag': True,
    'valid_steps': 2000,
    'warmup_steps': 1000,
    # 'total_steps': 70000, # len(train) * n_epochs
    'learning_rate': 1e-3,          
    'early_stop': 300,
    'n_workers': 8,
    'save_path': './models/model.ckpt'
}
print(device)
all_seed(config['seed'])


# # **导入数据集**
# - 将数据集分割成训练集(90%)和验证集(10%).
# - 创建dataloader用于模型训练.
# - 用`pad_sequence`方法将一个batch中的数据都扩展成一样的长度(`collate_batch`)  
# 
#     Example:
# ```python
# from torch.nn.utils.rnn import pad_sequence
# a = torch.ones(25, 40)
# b = torch.ones(22, 40)
# c = torch.ones(15, 40)
# pad_sequence([a, b, c], batch_first=True).size() # 都扩展成一样长
# torch.Size([3, 25, 40])
# ```

# In[31]:


def collate_batch(batch):
    # 将一个batch中的数据合并
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # 为了保持一个batch内的长度都是一样的所有需要进行padding, 同时设置batch的维度是最前面的一维
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) 一个很小的值
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


data_dir = config['dataset_dir']
dataset = myDataset(data_dir)
speaker_num = dataset.get_speaker_number()
speaker2id = dataset.speaker2id
# 将数据拆分成训练集和验证集
trainlen = int(0.9 * len(dataset))
lengths = [trainlen, len(dataset) - trainlen]
trainset, validset = random_split(dataset, lengths)
testset = InferenceDataset(data_dir)

train_loader = DataLoader(
    trainset,
    batch_size=config['batch_size'],
    shuffle=True,
    drop_last=True,
    num_workers=config['n_workers'],
    pin_memory=True,
    collate_fn=collate_batch,
)

valid_loader = DataLoader(
    validset,
    batch_size=config['batch_size'],
    num_workers=config['n_workers'],
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
)


test_loader = DataLoader(
    testset,
    batch_size=1,
    num_workers=config['n_workers'],
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    collate_fn=inference_collate_batch,
)


# #  &#x1F4CC; **开始训练！**

# In[32]:


model = Classifier(
    input_dim=40,  # n_mel
    d_model=80,
    n_spks=600, 
    dropout=0.1
).to(device)
trainer(train_loader, valid_loader, model, config, device)


# # **测试并生成预测结果的csv**

# In[33]:


model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(config['save_path']))
model_best.eval()
mapping_path = Path(data_dir) / "mapping.json"
mapping = json.load(mapping_path.open())
pred_id = []
pred_final_cls = []
with torch.no_grad():
    for name, data in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        pred_id += name
        pred_final_cls += [mapping["id2speaker"][str(test_label[0])]]


# In[34]:


df = pd.DataFrame()
df["Id"] = pred_id
df["Category"] = pred_final_cls
df.to_csv("submission.csv",index = False)


# # 贡献者
# 
# 孙成超
# - Github: https://github.com/scchy
# - Email: hyscc1994@foxmail.com
