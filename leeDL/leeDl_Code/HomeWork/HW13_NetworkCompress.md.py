#!/usr/bin/env python
# coding: utf-8

# # Homework 13 - 神经网络压缩(`Network Compression`)
# 
# 作者: Liang-Hsuan Tseng (b07502072@ntu.edu.tw), modified from ML2021-HW13  
# 如果你有任何问题, 可以免费询问: ntu-ml-2022spring-ta@googlegroups.com  
# 
# [**HW13 PPT**](https://docs.google.com/presentation/d/1nCT9XrInF21B4qQAWuODy5sonKDnpGhjtcAwqa75mVU/edit#slide=id.p)

# ## noteBook 目录
# 
# * [Packages](#Packages) - 安转必要的一些包
# * [Configs](#Configs) - 实验的配置，你可以在这里更改一些超参数。
# * [Dataset](#Dataset) - 您需要了解的有关数据集的信息。
# * [Architecture_Design](#Architecture_Design) - 深度(`depthwise`)和逐点(`pointwise`)卷积示例以及一些有用的链接。  
# * [Knowledge_Distillation](#Knowledge_Distillation) - 在知识提炼中的KL离散损失和一些有用的链接。
# * [Training](#Training) - 从HW3修改的训练循环实现。
# * [Inference](#Inference) - 用训练产出的`student_best.ckpt`生成`submission.csv` 。

# # Packages

# In[1]:


get_ipython().system('pip install torchsummary')


# In[2]:


import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset # "ConcatDataset" 和 "Subset" 有可能使用
from torchvision.datasets import DatasetFolder, VisionDataset
from torchsummary import summary
# from tqdm.auto import tqdm
from tqdm import tqdm
import random

# 查看GPU
get_ipython().system('nvidia-smi')


# # Configs
# 
# 在本部分中，你可以指定一些变量和超参数作为配置。

# In[3]:


def all_seed(seed=6666, env=None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
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


# In[4]:


cfg = {
    'dataset_root': '../input/ml2022spring-hw13/food11-hw13',
    'save_dir': './outputs',
    'exp_name': "simple_baseline",
    'batch_size': 64,
    'lr': 5e-4,
    'seed': 20220013,
    'loss_fn_type': 'KD', # simple baseline: CE, medium baseline: KD.
    'weight_decay': 0, #1e-5,
    'grad_norm_max': 10,
    'n_epochs': 20, # 训练更多的步骤以通过中等基线(medium baseline).
    'patience': 300,
}


# In[5]:


# 设置随机种子
all_seed(cfg['seed'])
save_path = os.path.join(cfg['save_dir'], cfg['exp_name'])
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

log_path = f"{save_path}/log.txt"
if os.path.exists(log_path):
    os.system(f"rm {log_path}")
# 定义简单的日志方法
log_fw = open(log_path, 'a+') # 打开日志文件保存日志
def log(text):     # 定义一个日志记录函数来跟踪训练过程
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

log(cfg)  # 写入配置


# In[ ]:





# # Dataset
# 
# 在本次作业中我们使用 Food11 数据集, 和homework3数据集相似，不过数据上稍微做了一些调整. 数据集可以直接在kaggle中载入，或者通过链接下载。
# 
# ```shell
# # 从github下载数据 (大约 1.12G)
# !wget https://github.com/virginiakm1988/ML2022-Spring/raw/main/HW13/food11-hw13.tar.gz
# # 备份链接:
# !wget https://github.com/andybi7676/ml2022spring-hw13/raw/main/food11-hw13.tar.gz -O food11-hw13.tar.gz
# # !gdown '1ijKoNmpike_yjUw8SWRVVWVoMOXXqycj' --output food11-hw13.tar.gz
# 
# # 解压
# !tar -xzf ./food11-hw13.tar.gz 
# # !tar -xzvf ./food11-hw13.tar.gz # 可以查看解压进度
# ```
# 

# In[6]:


get_ipython().system('ls ../input/ml2022spring-hw13/food11-hw13')


# In[7]:


for dirname, _, filenames in os.walk('../input/ml2022spring-hw13/food11-hw13'):
    if len(filenames) > 0:
        print(f"{dirname}: {len(filenames)} files.")


# 下一步, 特殊的train/test数据集变换进行数据扩增  
# Torchvision 提供了很多实用的图像预处理`image preprocessing`方法，数据扩增`data augmentation`方法
# 
# 可以参考 [PyTorch官方文档-transforms](https://pytorch.org/vision/stable/transforms.html) 了解不同的transforms方法。

# In[8]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 定义 training/testing transforms
test_tfm = transforms.Compose([
    # 如果你正在使用提供的教师模型(teacher model)，则不建议修改此部分。
    # 下列的transform 方法是标准的，并且足以进行测试。
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_tfm = transforms.Compose([
    # 在这里增加一些有用的transform 或 数据扩增方法, 基于你在HW3中学习的经验
    transforms.Resize(256),  # 你可以修改这里
    transforms.CenterCrop(224), # 你可以修改这里, 但是要注意，给定教师模型(teacher model)的输入大小是224。
    # 因此，除了224之外的输入大小可能会降低模型性能。需要注意。
    transforms.RandomHorizontalFlip(), # 你可以修改这里.
    transforms.ToTensor(),
    normalize,
])


# In[9]:


class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super().__init__()
        self.path = path
        self.files = sorted([os.path.join(path, i) for i in os.listdir(path) if i.endswith('.jpg')])
        if files is not None:
            self.files = files
        print(f'One {path} sample', self.files[0])
        self.tfm = tfm
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.tfm(im)
        try:
            label = int(fname.split("/")[-1].split('_')[0])
        except:
            label = -1
        return im, label


# In[10]:


train_set = FoodDataset(os.path.join(cfg['dataset_root'], "training"), tfm=train_tfm)
train_loader = DataLoader(train_set,batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

val_set = FoodDataset(os.path.join(cfg['dataset_root'], "validation"), tfm=test_tfm)
val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)


# # &#x2728; Architecture_Design
# 
# 在这个作业中我们需要设计一个更小的网络，并使它表现的十分良好。显然，一个好的网络结构的设计是十分关键的。  
# 这里我们介绍深度`depthwise`和逐点`pointwise`卷积. 当涉及到网络压缩时， 这些变体的卷积架构设计是一些常见技术。
# 
# - `depthwise`:
#     - 一个kenerl对一个channel
#     - in_channel == out_channel
#     - 缺点：无法捕捉channel之间的关系
#     
# - `pointwise`:
#     - `kernel_size=1`
#     - 仅仅考虑channel之间的关系
#     
# - `depthwise` + `pointwise`
#     - 参数减少 $\frac{1}{O}+\frac{1}{K\times K}$ `O-输出channel, K-kernel大小`
# 
# ![dpdw](./HW13_pic/dwpw.png)

# In[11]:


# 示例：Depthwise and Pointwise Convlution
def dwpw_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels), # depthwise convolution
        nn.Conv2d(in_channels, out_channels, 1) # pointwise convolution
    )


# - 其他有用的方法
#     - [group convolution](https://www.researchgate.net/figure/The-transformations-within-a-layer-in-DenseNets-left-and-CondenseNets-at-training-time_fig2_321325862)(实际上`depthwise convolution`是一种特殊的`group convolution`)
#     - [SqueezeNet](https://arxiv.org/abs/1602.07360)
#     - [MobileNet](https://arxiv.org/abs/1704.04861)
#     - [ShuffleNet](https://arxiv.org/abs/1707.01083)
#     - [Xception](https://arxiv.org/abs/1610.02357)
#     - [GhostNet](https://arxiv.org/abs/1911.11907)
#  
# 在介绍了深度卷积和点卷积之后，让我们定义**学生网络`student network`框架**。在这里，我们有一个由一些具有深度和逐点卷积的规则卷积层形成的简单网络。通过这种方式，你可以进一步增加网络的深度或宽度。
# 
# <font color=darkred><b>TODO：修改成自己的网络框架</font></b>   

# In[12]:


# 在这里定义自己的 student network.
# 我们将使用你的student network来评估您的结果（包括总参数量）

class StudentNetOrg(nn.Module):
    def __init__(self):
        super(StudentNetOrg, self).__init__()
        # TODO: 修改成自己的网络框架
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(64, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            # 在这里，我们对各种输入大小采用全局平均。
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(100, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        # TODO: 修改成自己的网络框架
        self.cnn = nn.Sequential(
            dwpw_conv(3, 32, 3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            dwpw_conv(32, 32, 3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            dwpw_conv(32, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            dwpw_conv(64, 100, 3, stride=1, padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            # 在这里，我们对各种输入大小采用全局平均。
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(100, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def get_student_model():
    return StudentNet()


# 确定`student network`框架后, 需要使用`torchsummary`获取网络的信息和验证参数总数. 需要注意`student network`网络参数总量，  
# 网络参数的总量不能超过限制(`总参数（torchsummary中展示）<=100,000`).

# In[13]:


student_model= get_student_model()
student_model_org = StudentNetOrg()
print('**'*35)
print("[ StudentNetOrg ]")
summary(student_model_org,  (3, 224, 224), device='cpu')
print("\n")
print('**'*35)
print("[ StudentNet ]")
summary(student_model, (3, 224, 224), device='cpu')


# In[14]:


# 载入提供的教师模型 (restnet18 num_classes=11, test-acc ~= 89.9%)
teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)
# load_state dict
teach_ckpt_path = os.path.join(cfg['dataset_root'], "resnet18_teacher.ckpt")
teacher_model.load_state_dict(torch.load(teach_ckpt_path, map_location='cpu'))
summary(teacher_model, (3, 224, 224), device='cpu')


# #   &#x2728; Knowledge_Distillation
# 
# 既然我们有一个学习过的大模型，那就让它教另一个小模型吧。在实现中，让训练目标是大模型的预测，而不是实际标签
# 
# 
# **为什么这样能有效训练出小网络 ?**
# - 如果数据干净，那么大模型的预测可能会忽略带有错误标记的数据的噪声
# - 类之间可能存在一些关系，因此教师模型中的软标签可能会很有用。例如，数字8与6、9、0比1、7更相似
# 
# **如何实施训练 ?**
# - 损失函数定义
# $$\text{Loss} = \alpha T^2 \times KL(p||q) + (1-\alpha)\text{(Original Cross Entropy Loss)}$$
# 
#     - $\text{where p=softmax}(\frac{\text{student's logits}}{T})$
#     - $\text{where q=softmax}(\frac{\text{teacher's logits}}{T})$
#     
#     
# - 使用链接: [`pytorch docs of KLDivLoss with examples` Link](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
# - 原始论文: [`Distilling the Knowledge in a Neural Network` Link](https://arxiv.org/abs/1503.02531)

# <font color=darkred><b>TODO：参考上述的函数，结合`KL divergence Loss`和`CE Loss`完成损失函数的定义</font></b>    

# In[15]:


a = torch.randn(4)
a


# In[16]:


sft = nn.Softmax(dim=-1)
sft(a), sft(a/1.15), sft(a/0.5), 


# In[17]:


# 利用 KL divergence loss 实现知识蒸馏(know distillation)的损失函数 
def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.5, temperature=1.15):
    # temperature 越大越平滑
    # TODO: 
    kl_loss = torch.nn.KLDivLoss(reduction='mean', log_target=True)
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    sft = nn.Softmax(dim=-1)
    return alpha * temperature * temperature * kl_loss(sft(student_logits/temperature), sft(teacher_logits/temperature)) \
            + (1-alpha) * ce_loss(student_logits, labels)


# In[18]:


print("cfg['loss_fn_type']=", cfg['loss_fn_type'])
#  选择损失函数
if cfg['loss_fn_type'] == 'CE':
    loss_fn = nn.CrossEntropyLoss() # simple base line

if cfg['loss_fn_type'] == 'KD':
    loss_fn = loss_fn_kd

# 还可以自定义一些其他方法
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'device: {device}')
device = torch.device(device)
n_epochs = cfg['n_epochs']
patience = cfg['patience']


# #   &#x2728; Training
# 
# 实现简单基线的训练循环，可以随意修改。

# In[19]:


# 模型初始化，并将参数移入训练设备
student_model.to(device)
opt = torch.optim.Adam(student_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

# 初始化跟踪器, 这部分不是训练参数不需要修改
stale = 0
best_acc = 0.0

teacher_model.to(device)
teacher_model.eval() # MEDIUM BASELINE
for epoch in range(n_epochs):
    # ---------------- Training ----------------------
    # 在训练之前，确保模型是开启训练模式的
    student_model.train()
    # 记录训练过程的信息
    train_loss = []
    train_accs = []
    train_lens = []
    tq_bar = tqdm(train_loader)
    tq_bar.set_description(f"[ Train | Epoch {epoch+1:03d} / {n_epochs:03d} ]")
    for imgs, labels in tq_bar:
        imgs = imgs.to(device)
        labels = labels.to(device)
#         imgs = imgs.half() # 开启半精度。直接可以加快运行速度、减少GPU占用，并且只有不明显的accuracy损失。
        # 前向传播
        with torch.no_grad():  # MEDIUM BASELINE
            teacher_logits = teacher_model(imgs)  # MEDIUM BASELINE
        logits = student_model(imgs)
        # 计算损失.
        loss = loss_fn(logits, labels, teacher_logits)
#         loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (logits.argmax(dim=-1) == labels).float().sum()
        # 记录 loss 和 accuracy.
        batch_len = len(imgs)
        train_loss.append(loss.cpu().item() * batch_len)
        train_accs.append(acc)
        train_lens.append(batch_len)
        tq_bar.set_postfix({"loss" : np.mean(train_loss[-10:])})
    
    train_loss = sum(train_loss)/sum(train_lens)
    train_acc = sum(train_accs)/sum(train_lens)
    # 打印信息
    log(f'[ Train | {epoch+1:03d} / {n_epochs:03d} ] loss = {train_loss:.5f} acc = {train_acc:.5f}')
    # ---------------- validation ----------------------
    student_model.eval()
    val_loss = []
    val_accs = []
    val_lens = []
    tq_bar = tqdm(val_loader)
    tq_bar.set_description(f"[ Val | Epoch {epoch+1:03d} / {n_epochs:03d} ]")
    for imgs, labels in tq_bar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        # 前向传播
        with torch.no_grad():  # MEDIUM BASELINE
            teacher_logits = teacher_model(imgs)  # MEDIUM BASELINE
        with torch.no_grad():
            logits = student_model(imgs)
        loss = loss_fn(logits, labels, teacher_logits)
#         loss = loss_fn(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().sum()
        # 记录 loss 和 accuracy.
        batch_len = len(imgs)
        val_loss.append(loss.cpu().item() * batch_len)
        val_accs.append(acc)
        val_lens.append(batch_len)
        tq_bar.set_postfix({"loss" : np.mean(val_loss[-10:])})
    
    val_loss = sum(val_loss)/sum(val_lens)
    val_acc = sum(val_accs)/sum(val_lens)
    log(f'[ Val | {epoch+1:03d} / {n_epochs:03d} ] loss = {val_loss:.5f} acc = {val_acc:.5f}')
    # 更新logs
    if val_acc > best_acc:
        log(f'Best model found at epoch {epoch+1}. saving model. acc={val_acc:.5f}')
        best_acc = val_acc
        torch.save(student_model.state_dict(), f"{save_path}/student_best.ckpt")
        stale = 0
    else:
        stale += 1
        if (stale > patience):
            log(f'No improving {patience} consecutions. early stopping')
            break
    
log("Finish training")
log_fw.close()


# # Inference
# 载入训练好的最佳模型进行预测并生成`submission.csv`

# In[20]:


eval_set = FoodDataset(os.path.join(cfg['dataset_root'], "evaluation"), tfm=test_tfm)
eval_loader = DataLoader(eval_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)


# In[21]:


# 载入模型
student_model_best = get_student_model()
ckpt_path = f"{save_path}/student_best.ckpt" 
student_model_best.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
student_model_best.to(device) 

# 开始评估
student_model_best.eval()
eval_preds = [] # storing predictions of the evaluation dataset

for imgs, _ in tqdm(eval_loader):
    # 在eval中不需要进行梯度下降
    with torch.no_grad():
        logits = student_model_best(imgs.to(device))
        preds = list(logits.argmax(dim=-1).squeeze().cpu().numpy())

    eval_preds += preds

def pad4(i):
    return str(i).zfill(4)

# 保存结果
ids = [pad4(i) for i in range(0, len(eval_set))]
categories = eval_preds

df = pd.DataFrame()
df['Id'] = ids
df['Category'] = categories
df.to_csv(f"submission.csv", index=False) 


# In[ ]:




