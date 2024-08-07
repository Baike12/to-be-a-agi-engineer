#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **作业 1: COVID-19 病例预测 (回归)**

# 学习目标:
# * 模型拟合：用深度神经网络 (DNN)拟合一个回归模型。
# * 掌握技巧：
#   * 熟悉并掌握DNN模型训练的基本技巧。
#   * 提高PyTorch的使用熟练度。

# #  数据集下载
# 如果下列命令无法下载，可以到下列地址下载数据
# - Kaggle下载数据:
# [Kaggle: ml2022spring-hw1](https://www.kaggle.com/competitions/ml2022spring-hw1)
# - 百度云下载数据: [云盘(提取码：ml22)](https://pan.baidu.com/s/1ahGxV7dO2JQMRCYbmDQyVg)
# 
# <b>同学下载完后直接解压到 HW01文件夹下面（将里面的文件最终放到HW01下）</b>
# 

# In[ ]:


get_ipython().system("gdown --id '1kLSW_-cW2Huj7bh84YTdimGBOJaODiOS' --output covid.train.csv")
get_ipython().system("gdown --id '1iiI5qROrAhZn-o4FPqsE97bMzDEFvIdg' --output covid.test.csv")


# # 导入包

# In[1]:


# 数值、矩阵操作
import math
import numpy as np

# 数据读取与写入make_dot
import pandas as pd
import os
import csv

# 进度条
from tqdm import tqdm
# 如果是使用notebook 推荐使用以下（颜值更高 : ) ）
# from tqdm.notebook import tqdm

# Pytorch 深度学习张量操作框架
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# 绘制pytorch的网络
from torchviz import make_dot

# 学习曲线绘制
from torch.utils.tensorboard import SummaryWriter


# # 一些功能函数
# 
# 一些重要的方法（随机种子设置、数据拆分、模型预测）  
# <b>这部分可以不做修改</b>
# 
# 

# In[2]:


def same_seed(seed): 
    '''
    设置随机种子(便于复现)
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set Seed = {seed}')


def train_valid_split(data_set, valid_ratio, seed):
    '''
    数据集拆分成训练集（training set）和 验证集（validation set）
    '''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval() # 设置成eval模式.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():
            pred = model(x)         
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds


# # 数据集

# In[3]:


class COVID19Dataset(Dataset):
    '''
    x: np.ndarray  特征矩阵.
    y: np.ndarray  目标标签, 如果为None,则是预测的数据集
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


# #  &#x2728; 神经网络模型
# 
# 可以在以下My_Model类框架下，进行不同结构的深度模型尝试

# In[4]:


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: 修改模型结构, 注意矩阵的维度（dimensions） 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x


# #  &#x2728;  特征选择
# <font color=darkred><b>***TODO***: 通过修改下面的函数，选择自己认为有用的特征</font></b>

# In[5]:


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    特征选择
    选择较好的特征用来拟合回归模型
    '''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: 选择需要的特征 ，这部分可以自己调研一些特征选择的方法并完善.

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid


# #   &#x2728; 训练部分
# <font color=darkred><b>***TODO1***: 可以查看学习, 尝试更多的优化器 https://pytorch.org/docs/stable/optim.html </font></b>  
# <font color=darkred><b>***TODO2***:  L2 正则( 可以使用optimizer(weight decay...) 或者 自己实现L2正则</font></b>  
# 
# 
# 
#     
# - 查看学习更多的优化器:
#     - [pytorch官网optim](https://pytorch.org/docs/stable/optim.html)
#     - 也可以搜索一些优化器资料
#         - 比如[momentum](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)
#         - 比如[华盛顿大学-sgd_averaging](https://courses.cs.washington.edu/courses/cse547/18sp/slides/sgd_averaging.pdf)
#         - 比如[nesterov](https://cs231n.github.io/neural-networks-3/#sgd)
# 
# 
# ------
# ```text
# torch.optim.SGD(
#      params (iterable):  待优化参数的iterable( model.parameters() )或者是定义了参数组的dict
#      lr (float, 必填): 学习率
#      momentum (float, 可选): 动量权重（默认：0）
#      weight_decay (float, 可选): 权重衰减（L2惩罚）（默认: 0）
#      dampening (float, 可选): 抑制momentum，越大动量越小 (默认: 0)
# ```
# 从下列动量迭代公式可以看出: $\textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t$
# 
# ```text
#      nesterov (bool, 可选): 是否采用 Nesterov momentum (默认: False)
#          其核心思想是，计算“预览”位置的梯度而不是当前位置的梯度。计算梯度之前，由于动量项的作用是轻推参数向量mu*v，可以估计未来的位置为x+mu*v，这个位置在接下来实际达到位置的附近。用该位置计算梯度，并修正动量。
#      maximize (bool, 可选): 是否基于目标函数最大化参数, 反之最小化 (默认: False 即梯度下降)
#  )
# ```
# ------
# SGD算法:  
# \begin{aligned}
#       &\rule{110mm}{0.4pt}                                                                 \\
#       &\textbf{input}      : \gamma \text{ (lr-学习率)}, \: \theta_0 \text{ (params-优化参数)}, \: f(\theta)
#           \text{ (objective-目标函数)}, \: \lambda \text{ (weight decay-L2正则参数)},                          \\
#       &\hspace{13mm} \:\mu \text{ (momentum-动量权重)}, \:\tau \text{ (dampening-动量抑制权重)},
#       \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
#       &\rule{110mm}{0.4pt}                                                                 \\
#       &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
#       &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
#       &\hspace{5mm}\textbf{if} \: \lambda \neq 0  \ \ \  \text{ 如果L2正则参数不为0, 导数中增加L2正则项的偏导}                   \\
#       &\hspace{10mm} \frac{\partial (\frac{1}{2}\theta_{t-1}^2)}{\partial\theta_{t-1} } = \theta_{t-1} \ \ \  \text{L2正则项的偏导}  \\
#       &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                          \\
#       &\hspace{5mm}\textbf{if} \: \mu \neq 0  \ \ \  \text{ 如果动量不为0}      \\
#       &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
#       &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
#       &\hspace{10mm}\textbf{else}                                                          \\
#       &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
#       &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
#       &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
#       &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
#       &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
#       &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
#       &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
#       &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
#       &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
#       &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
#       &\bf{return} \:  \theta_t                                                     \\[-1.ex]
#       &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
#  \end{aligned}
# 

# In[6]:


def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # 损失函数的定义

    # 定义优化器
    # TODO: 可以查看学习更多的优化器 https://pytorch.org/docs/stable/optim.html 
    # TODO: L2 正则( 可以使用optimizer(weight decay...) )或者 自己实现L2正则.
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 
    
    # tensorboard 的记录器
    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        # 创建文件夹-用于存储模型
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # 训练模式
        loss_record = []

        # tqdm可以帮助我们显示训练的进度  
        train_pbar = tqdm(train_loader, position=0, leave=True)
        # 设置进度条的左边 ： 显示第几个Epoch了
        train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
        for x, y in train_pbar:
            optimizer.zero_grad()               # 将梯度置0.
            x, y = x.to(device), y.to(device)   # 将数据一到相应的存储位置(CPU/GPU)
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # 反向传播 计算梯度.
            optimizer.step()                    # 更新网络参数
            step += 1
            loss_record.append(loss.detach().item())
            
            # 训练完一个batch的数据，将loss 显示在进度条的右边
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        # 每个epoch,在tensorboard 中记录训练的损失（后面可以展示出来）
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # 将模型设置成 evaluation 模式.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # 每个epoch,在tensorboard 中记录验证的损失（后面可以展示出来）
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # 模型保存
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


# # 超参设置
# `config` 包含所有训练需要的超参数（便于后续的调参），以及模型需要存储的位置

# In[7]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # 随机种子，可以自己填写. :)
    'select_all': True,   # 是否选择全部的特征
    'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'n_epochs': 3000,     # 数据遍历训练次数           
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # 如果early_stop轮损失没有下降就停止训练.     
    'save_path': './models/model.ckpt'  # 模型存储的位置
}


# # 导入数据集
# 1. 从文件中读取数据`pd.read_csv`
# 2. 数据拆分成三份 训练（training）、验证（validation）、测试（testing）
#     - `train_valid_split`：  分成训练、验证
#     - `select_feat`：拆分特征和label，并进行特征选择
#     - `COVID19Dataset`: 分别将训练、验证、测试集的特征和label组合成可以用于快速迭代训练的数据集`train_dataset, valid_dataset, test_dataset`
# 
# <b>这部分不用修改</b>

# In[8]:


# 设置随机种子便于复现
same_seed(config['seed'])


# 训练集大小(train_data size) : 2699 x 118 (id + 37 states + 16 features x 5 days) 
# 测试集大小(test_data size）: 1078 x 117 (没有label (last day's positive rate))
pd.set_option('display.max_column', 200) # 设置显示数据的列数
train_df, test_df = pd.read_csv('./covid.train.csv'), pd.read_csv('./covid.test.csv')
display(train_df.head(3)) # 显示前三行的样本
train_data, test_data = train_df.values, test_df.values
del train_df, test_df # 删除数据减少内存占用
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# 打印数据的大小
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# 特征选择
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# 打印出特征数量.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# 使用Pytorch中Dataloader类按照Batch将数据集加载
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


# #  &#x1F4CC;开始训练！

# In[9]:


model = My_Model(input_dim=x_train.shape[1]).to(device) # 将模型和训练数据放在相同的存储位置(CPU/GPU)
trainer(train_loader, valid_loader, model, config, device)


# # 使用 `tensorboard` 输出模型训练过程和指标可视化(可选)
# 
# `tensorboard` 可视化工具：可以记录并展现模型的训练过程中的各种指标，这里我们是记录模型的损失
# 
# 运行下面代码需要等待几分钟，因为需要时间加载记录的数据并绘制学习曲线。如果还是没有展示出来，可以再运行一下试试。

# In[11]:


get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir=./runs/ --port=6007')


# ![tensorBoard](./pic/tensorBoard.png)

# # 测试部分
# 测试集的预测结果保存到`pred.csv`.

# In[12]:


def save_pred(preds, file):
    ''' 将模型保存到指定位置'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 
save_pred(preds, 'pred.csv')         


# # 参考文献
# 本文件所使用&参考的代码来源于Heng-Jui Chang @ NTUEE (https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)

# # 贡献者
# 
# 孙成超
# - Github: https://github.com/scchy
# - Email: hyscc1994@foxmail.com
