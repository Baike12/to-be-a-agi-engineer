#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **作业 14: 终生机器学习 `LifeLong Machine Learning`**
# ### 助教的PPT
# [PPT](https://docs.google.com/presentation/d/1SMJLWPTPCIrZdNdAjrS4zQZx1kfB73jCFSb7JRX90gQ/edit?usp=sharing)
# 
# - 终生学习的定义
#   - 关于终身学习的详细解释和定义，请参阅 [LifeLong learning | youtu.be](https://youtu.be/7qT5P9KJnWo) 
# 
# - 终生学习的一些方法
#   - 有人在2019年底为LifeLong Learning提出了一份调查论文，将2016-2019年的LigeLong学习方法分为三个家族。
# 依据任务数据信息在整个学习过程中的存储和使用方式的差异，可以将LifeLong学习方法分为三类：
#     * 基于重采样的方法   `Replay-based methods`
#     * 基于正则化的方法  `Regularization-based methods`
#     * 参数隔离方法      `Parameter isolation methods`
# 
# ![LL_summary](./HW14_pic/LLL_summary.png)
# 
# 在本次作业中，我们关注**基于正则化的方法**中的`prior-focused`方法: **EWC、MAS、SI、Remanian Walk、SCP**
# 
# 
# 
# 论文: [Continual Learning in Neural
# Networks](https://arxiv.org/pdf/1910.02718.pdf)
# 
# 如果您有任何问题，请随时给我们发邮件。[ntu-ml-2022spring-ta@googlegroups.com](ntu-ml-2022spring-ta@googlegroups.com)

# # 导入包

# In[1]:


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import random


# In[2]:


get_ipython().system('nvidia-smi')


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

all_seed(0)


# # 数据准备
# 
# 我们使用旋转的手写识别数据集MNIST(`rotated MNIST`) 作为训练集  
# 因此，我们使用5种不同的旋转来生成5种不同的MNIST作为不同的任务
# 
# ## 数据旋转与转换

# In[4]:


# 旋转 MNIST 数据生成 5 个任务
def _rotate_image(image, angle):
    if angle is None:
        return image
    image = transforms.functional.rotate(image, angle=angle)
    return image


def get_transform(angle=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
       transforms.Lambda(lambda x: _rotate_image(x, angle)),
       Pad(28)
    ])
    return transform


class Pad(object):
    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # 输入图片的长宽是一致的，如果输入图片的size比期望的size小时，对图片的四周进行扩充
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)

    
class Data():
    def __init__(self, path, train=True, angle=None):
        transform = get_transform(angle)
        self.dataset = datasets.MNIST(root=os.path.join(path, "MNIST"), transform=transform, train=train, download=True)


# ## 构建Dataloaders数据及训练参数
# - 一些训练参数
#   - 设置 5 个不同的旋转
#   - 5 Train DataLoader
#   - 5 Test DataLoader 

# In[5]:


class Args:
    task_number = 5
    epochs_per_task = 10
    lr = 1.0e-4
    batch_size = 128
    test_size=8192

args=Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 给每个task 生成旋转角度.
angle_list = [20 * x for x in range(args.task_number)]
# 准备5个Task: 不同旋转角度的 MNIST 数据集.
train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]

test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets]


# ### 可视化

# In[6]:


# 分别绘制5个任务中的0-9图像
sample = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]

plt.figure(figsize=(30, 10))
for task in range(5):
    target_list = []
    cnt = 0
    while (len(target_list) < 10):
        img, target = sample[task].dataset[cnt]
        cnt += 1
        if target in target_list:
            continue
        else:
            target_list.append(target)
        plt.subplot(5, 10, (task)*10 + target + 1)
        curr_img = np.reshape(img, (28, 28))
        plt.matshow(curr_img, cmap=plt.get_cmap('gray'), fignum=False)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title("task: " + str(task+1) + " " + "label: " + str(target), y=1)


# # 模型准备 
# ## 模型架构
# 
# 为了公平地比较，采用同一模型进行序列任务连续训练，该模型体系结构由4层全连接层组成。

# In[7]:


class Model(nn.Module):
    """
    模型架构
    1*28*28 (input) → 1024 → 512 → 256 → 10
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1*28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


example = Model()
print(example)


# # 训练与评估
# 
# ## 模型训练
# 以下是我们模型训练的fuction，能同样适用于本次作业中不同的终身学习算法（基于正则化）的训练

# In[8]:


def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device, log_step=1):
    model.train()
    model.zero_grad()
    objective = nn.CrossEntropyLoss()
    acc_per_epoch = []
    loss = 1.0
    for epoch in range(epochs_per_task):
        tr_bar = tqdm(dataloader, leave=False)
        for imgs, labels in  tr_bar: 
            tr_bar.set_description(f"[ Train | Epoch {epoch+1:02d}/{epochs_per_task:02d} ]")
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = objective(outputs, labels)
            total_loss = loss
            # ----------  正则化 ----------
            lll_loss = lll_object.penalty(model)
            total_loss += lll_lambda * lll_loss 
            # model update
            lll_object.update(model)
            # ----------------------------
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = total_loss.item()
            tr_bar.set_postfix({"Loss": f"{loss:.7f}"})

        acc_average  = []
        for test_dataloader in test_dataloaders: 
            acc_test = evaluate(model, test_dataloader, device)
            acc_average.append(acc_test)
        average = np.mean(np.array(acc_average))
        acc_per_epoch.append(average*100.0)
        print(f"[ Val | Epoch {epoch+1:02d}/{epochs_per_task:02d} ] acc={average*100:.3f} %")
    return model, optimizer, acc_per_epoch


# ## 模型评估
# 以下是我们模型评估的fuction，能同样适用于本次作业中不同的终身学习算法（基于正则化）的评估
# 

# In[9]:


def evaluate(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total = 0
    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()
    return correct_cnt / total


# # &#x2728; 终生学习——基于正则化的`prior-focused`方法
# - Baseline
# - EWC
# - SI
# - MAS
# - RWalk
# - SCP

# In[10]:


class baseline(object):
    """
    基准`baseline`方法 : 不做正则化 [ 初始化权重矩阵全为0 ]
    """
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # 提取模型中的所有参数
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} 
        # 保存当前参数
        self.p_old = {} 
        # 生成权重矩阵-全都为0
        self._precision_matrices = self._calculate_importance()  
        for n, p in self.params.items():
            # 将原始参数保存在 p_old 中- 保存为不可导
            self.p_old[n] = p.clone().detach() 

    def _calculate_importance(self):
        precision_matrices = {} 
        # 初始化权重（fill zero）
        for n, p in self.params.items(): 
            precision_matrices[n] = p.clone().detach().fill_(0)

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss

    def update(self, model):
        # do nothing
        return


# In[12]:


# Baseline
print("RUN BASELINE")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=baseline(model=model, dataloader=None, device=device)
lll_lambda=0.0
baseline_acc=[]
task_bar = tqdm(range(len(train_dataloaders)))

# 不断地对每项任务进行迭代训练
for train_indexes in task_bar:
    task_bar.set_description(f"Task  {train_indexes+1:02d}")
    print(f'train_indexes={train_indexes}\n', '--'*25)
    # 训练每个任务
    model, _, acc_list = train(
        model, optimizer, train_dataloaders[train_indexes], 
        args.epochs_per_task, 
        lll_object, lll_lambda, 
        evaluate=evaluate,
        device=device, 
        # 评估 所有训练过的task对应的test数据
        test_dataloaders=test_dataloaders[:train_indexes+1])

    # 获取模型权重（对于baseline 实际无任何变化） 
    lll_object=baseline(model=model, dataloader=train_dataloaders[train_indexes],device=device)
    # 使用新的优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 收集每个epoch的平均精度
    baseline_acc.extend(acc_list)


print(baseline_acc)
print("==================================================================================================")


# ## &#x2728;  EWC
# > 输入 新task的训练数据和`label`
# 
# 弹性权重合并(`Elastic Weight Consolidation`)
# 
# `ewc` 类中实现了`EWC`算法去计算正则项。这个核心概念可以看李宏毅老师的课程。这里我们将重点讨论EWC的算法。  
# 在这项作业中，我们想让我们的模型依次学习10项任务。这里我们展示了一个简单的例子，让模型依次学习2个任务（任务a和任务B）。
# 
# 在 EWC 算法中, 损失函数定义如下:
#  $$\mathcal{L}_B = \mathcal{L}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_{i} - \theta_{A,i}^{*})^2  $$
#   
# 假设我们有一个具有两个以上参数的神经网络。 $F_i$对应于李宏毅教授讲座中的$i^{th}$守卫. 请不要修改此参数，因为它对任务A很重要。
# 
# $F$定义如下:  **即梯度的平方**
# $$ F = [ \nabla \log(p(y_n | x_n, \theta_{A}^{*})) \nabla \log(p(y_n | x_n, \theta_{A}^{*}))^T ] $$ 
# 
# 我们只取矩阵的对角线值来近似每个参数的$F_i$.
# 
# 详细信息和推导在[Continual Learning in Neural
# Networks](https://arxiv.org/pdf/1910.02718.pdf) 中 2.4.1 and 2.4 
# 
# 参考论文: [Elastic Weight Consolidation](https://arxiv.org/pdf/1612.00796.pdf)

# In[13]:


class ewc(object):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """
    def __init__(self, model, dataloader, device, prev_guards=[None]):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # 提取模型中的所有参数
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} 
        self.p_old = {}
        # 保存之前的 guards
        self.previous_guard_list = prev_guards
        # 生成EWC的权重Fisher (F)矩阵， 即 F 函数
        self._precision_matrices = self._calculate_importance()
        for n, p in self.params.items():
            # 保留原始数据 - 保存为不可导
            self.p_old[n] = p.clone().detach()
        
    def _calculate_importance(self):
        out = {}
        # 初始化 Fisher (F) 矩阵（全部填充0）并加上之前的 guards
        for n, p in self.params.items():
            out[n] = p.clone().detach().fill_(0)
            for prev_guard in self.previous_guard_list:
                if prev_guard:
                    out[n] += prev_guard[n]

        self.model.eval()
        if self.dataloader is not None:
            number_data = len(self.dataloader)
            for x, y in self.dataloader:
                self.model.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                # 为EWC 生成 Fisher(F) 矩阵
                loss = F.nll_loss(F.log_softmax(pred, dim=1), y)
                loss.backward()
            
                for n, p in self.model.named_parameters():
                    # 获取每个参数的样本平均 梯度平方
                    out[n].data += p.grad.data ** 2 / number_data
                    
        out = {n: p for n, p in out.items()}
        return out

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # 最终的正则项 =   ewc权重 * 权重变化平方((p - self.p_old[n]) ** 2) 
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss
  
    def update(self, model):
        return 


# In[14]:


# EWC
print("RUN EWC")
model = Model()
model = model.to(device)
# 初始化优化器
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lll_object=ewc(model=model, dataloader=None, device=device)
# 设置正则项系数
lll_lambda=100
ewc_acc= []
task_bar = tqdm(range(len(train_dataloaders)))
prev_guards = []
for train_indexes in task_bar:
    task_bar.set_description(f"Task  {train_indexes+1:02d}")
    print(f'train_indexes={train_indexes}\n', '--'*25)
    model, _, acc_list = train(
        model, optimizer, train_dataloaders[train_indexes], 
        args.epochs_per_task, lll_object, lll_lambda, 
        evaluate=evaluate,device=device, 
        # 评估 所有训练过的task对应的test数据
        test_dataloaders=test_dataloaders[:train_indexes+1])

    # 获取模型权重 并 计算每个权重的ewc-guidance权重
    prev_guards.append(lll_object._precision_matrices)
    lll_object=ewc(model=model, dataloader=train_dataloaders[train_indexes], device=device, prev_guards=prev_guards)

    # 使用新的优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 收集每个epoch的平均精度
    ewc_acc.extend(acc_list)


print(ewc_acc)
print("==================================================================================================")


# ## &#x2728; MAS
# > 输入 新task的训练数据
# 
# <font color=darkred><b>TODO: 基于论文公式编写 Omega(Ω) 矩阵的生成</font></b>
# 
# 
# 记忆感知突触(`Memory Aware Synapses`)  
# `mas` 类中实现了`MAS`算法去计算正则项。
# 
# `MAS` 的核心和`EWC`相似, 唯一的区别是重要权重（`important weight`）的计算。  
# 以下提到了详细信息。
# 
# MAS:
# 
# 在`MAS`中损失函数如下, 模型在学习任务B之前学习任务A。
# 
# $$\mathcal{L}_B = \mathcal{L}(\theta) + \sum_{i} \frac{\lambda}{2} \Omega_i (\theta_{i} - \theta_{A,i}^{*})^2$$
# 
# 相对`EWC`来说, 在损失函数中 $F_i$ 被 $\Omega_i$ 替代，$\Omega_i$ 计算方法如下
# 
# $$\Omega_i = || \frac{\partial \ell_2^2(M(x_k; \theta))}{\partial \theta_i} || $$ 
# 
# $x_k$ 是之前任务中的样本数据。所以 $\Omega$是所学习的网络输出的平方L2范数的梯度。  
# 论文中提出的方法是通过从模型的每一层获取平方L2范数输出的局部版本。
# 在这里，我们只需要你通过模型的最后一层获取输出来实现全局版本。
# 
# 参考论文: 
# [Memory Aware Synapses](https://arxiv.org/pdf/1711.09601.pdf)
# 
# 

# In[15]:


class mas(object):
    """
    @article{aljundi2017memory,
        title={Memory Aware Synapses: Learning what (not) to forget},
        author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
        booktitle={ECCV},
        year={2018},
        url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """
    def __init__(self, model, dataloader, device, prev_guards=[None]):
        self.model = model 
        self.dataloader = dataloader
        # 提取模型全部参数
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} 
        # 参数初始化
        self.p_old = {} 
        self.device = device
        # 保存之前的 guards
        self.previous_guards_list = prev_guards
        # 生成 Omega(Ω) 矩阵
        self._precision_matrices = self._calculate_importance() 
        for n, p in self.params.items():
            # 保留原始数据 - 保存为不可导
            self.p_old[n] = p.clone().detach()

    def _calculate_importance(self):
        out = {}
        # 初始化 Omega(Ω) 矩阵（全部填充0）并加上之前的 guards
        for n, p in self.params.items():
            out[n] = p.clone().detach().fill_(0)
            for prev_guard in self.previous_guards_list:
                if prev_guard:
                    out[n] += prev_guard[n]

        self.model.eval()
        if self.dataloader is not None:
            number_data = len(self.dataloader)
            for x, y in self.dataloader:
                self.model.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                ################################################################
                #####  TODO: 生成 Omega(Ω) 矩阵.  #####   
                ################################################################
                # 网络输出 L2范数平方的梯度
                loss = torch.mean(torch.sum(pred ** 2, axis=1))
                loss.backward()
                for n, p in self.model.named_parameters():
                    out[n].data += torch.sqrt(p.grad.data ** 2) / number_data
                ################################################################      

        out = {n: p for n, p in out.items()}
        return out

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # 最终的正则项 =   Omega(Ω)权重 * 权重变化平方((p - self.p_old[n]) ** 2) 
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss
  
    def update(self, model):
        return 


# In[16]:


# MAS
print("RUN MAS")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=mas(model=model, dataloader=None, device=device)
lll_lambda=0.1
mas_acc= []
task_bar = tqdm(range(len(train_dataloaders)))
prev_guards = []

for train_indexes in task_bar:
    task_bar.set_description(f"Task  {train_indexes+1:02d}")
    print(f'train_indexes={train_indexes}\n', '--'*25)
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, 
                               evaluate=evaluate,device=device, 
                               # 评估 所有训练过的task对应的test数据
                               test_dataloaders=test_dataloaders[:train_indexes+1])

    # 获取模型权重 并 计算每个权重的 mas-guidance权重
    prev_guards.append(lll_object._precision_matrices)
    lll_object=mas(model=model, dataloader=train_dataloaders[train_indexes], device=device, prev_guards=prev_guards)

    # 使用新的优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 收集每个epoch的平均精度
    mas_acc.extend(acc_list)


print(mas_acc)
print("==================================================================================================")


# ## &#x2728; SI
# 
# `si`类中实现了 突触智能 (`Synaptic Intelligence`)算法去计算正则项
# 
# 在`SI`中损失函数如下, 模型在学习任务B之前学习任务A。
# 
# $$\mathcal{L}_B = \mathcal{L}(\theta) +  c \sum_{i}\Omega_i (\theta_{i} - \theta_{A,i}^{*})^2$$
# 
# $\Omega_i $ 计算如下：
# $$\Omega_i = \frac{\mathcal{w}_i}{(\Delta _i)^2 + \epsilon} $$ 
# - $\int _C \mathcal{g}(\theta (t))d\theta= \int_{t_0}^{t_1}\mathcal{g}(\theta (t)) \dot \theta^{'}(t)dt=-\sum_k\mathcal{w}_k$
#     - 简单理解就是 `-W`权重等于模型当前梯度$\mathcal{g(\theta (t))}$ 乘以 参数前后变化$\theta^{'}(t)$ 
#     - code: `self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))`
# 
# 参考论文：
# [Continual Learning Through Synaptic Intelligence](https://arxiv.org/pdf/1703.04200.pdf)

# In[17]:


# SI
class si(object):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """
    def __init__(self, model, dataloader, epsilon, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.epsilon = epsilon
        # 提取模型全部参数
        self.params = {n.replace('.', '__'): p for n, p in self.model.named_parameters() if p.requires_grad}
        # 计算权重
        self._n_p_prev, self._n_omega = self._calculate_importance() 
        self.W, self.p_old = self._init_()
    
    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.params.items():
            W[n] = p.data.clone().zero_()
            p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}
        
        for n, p in self.params.items():
            if self.dataloader is not None: # dataloader的作用是提示是否已经训练过
                # 查找/计算参数二次惩罚（quadratic penalty）的新值
                p_prev = getattr(self.model, f'{n}_SI_prev_task')
                W = getattr(self.model, f'{n}_W')
                p_cur = p.detach().clone()
                delta = p_cur - p_prev
                omega_add = W / (delta ** 2 + self.epsilon)
                try:
                    omega = getattr(self.model, f'{n}_SI_omega')
                except AttributeError:
                    omega = p.detach().clone().zero_()
                
                omega_new = omega + omega_add
                n_omega[n] = omega_new
                n_p_prev[n] = p_cur
                
                # 保存新参数到模型中
                self.model.register_buffer(f'{n}_SI_prev_task', p_cur)
                self.model.register_buffer(f'{n}_SI_omega', omega_new)
                continue

            n_p_prev[n] = p.detach().clone()
            n_omega[n] = p.detach().clone().zero_()
            self.model.register_buffer(f'{n}_SI_prev_task', p.detach().clone())
        return n_p_prev, n_omega

    def penalty(self, model):
        loss = 0.0 
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                _loss = self._n_omega[n] * (p - self._n_p_prev[n]) ** 2
                loss += _loss.sum()
        return loss
    
    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    #  - \sum_k grad *  \theta (t)'
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer(f'{n}_W', self.W[n])
                self.p_old[n] = p.detach().clone()
        return 


# In[18]:


# SI
print("RUN SI")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=si(model=model, dataloader=None, epsilon=0.1, device=device)
lll_lambda=1
si_acc = []
task_bar = tqdm(range(len(train_dataloaders)))

for train_indexes in task_bar:
    task_bar.set_description(f"Task  {train_indexes+1:02d}")
    print(f'train_indexes={train_indexes}\n', '--'*25)
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, 
                               evaluate=evaluate,device=device, 
                               test_dataloaders=test_dataloaders[:train_indexes+1])

    # 获取模型权重 并 计算每个权重的 SI-guidance权重
    lll_object=si(model=model, dataloader=train_dataloaders[train_indexes], epsilon=0.1, device=device)

    # 使用新的优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 收集每个epoch的平均精度
    si_acc.extend(acc_list)
    

print(si_acc)
print("==================================================================================================")


# ## &#x2728; RWalk
# 增量学习的Remanian Walk
# 
# `rwalk`类中实现了`Remanian Walk`算法去计算正则项
# 
# <font color=darkred><b>结合了ewc和SI</b></font>
# 
# 参考论文：
# [Riemannian Walk for Incremental Learning](https://arxiv.org/abs/1801.10112)

# In[19]:


class rwalk:
    def __init__(self, model, dataloader, epsilon, device, prev_guards=[None]):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.epsilon = epsilon
        self.update_ewc_parameter = 0.4
        # 提取模型全部参数
        self.params = {n.replace('.', '__'): p for n, p in self.model.named_parameters() if p.requires_grad}
        # 初始化 guidance martix
        self._means = {} 
        self.previous_guards_list = prev_guards
        
        # 生成 ewc-Fisher (F) 信息矩阵
        self._precision_matrices = self._calculate_importance_ewc()
        
        self._n_p_prev, self._n_omega = self._calculate_importance() 
        self.W, self.p_old = self._init_()

    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.params.items():
            W[n] = p.data.clone().zero_()
            p_old[n] = p.data.clone()
        return W, p_old
    
    def _calculate_importance_ewc(self):
        out = {}
        # 初始化 Fisher (F) 矩阵（全部填充0）并加上之前的 guards
        for n, p in self.params.items():
            out[n] = p.clone().detach().fill_(0)
            for prev_guard in self.previous_guards_list:
                if prev_guard:
                    out[n] += prev_guard[n]

        self.model.eval()

        if self.dataloader is not None:
            number_data = len(self.dataloader)                
            for x, y in self.dataloader:
                self.model.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                # 为EWC 生成 Fisher(F) 矩阵
                loss = F.nll_loss(F.log_softmax(pred, dim=1), y)
                loss.backward()                                                                                                     
                                                                                                                                                    
                for n, p in self.model.named_parameters():                                                 
                    n = n.replace('.', '__')
                    # 和 ewc 不一样的是进行了软更新
                    out[n].data = (1 - self.update_ewc_parameter) * out[n].data + self.update_ewc_parameter * p.grad.data ** 2 / number_data    
                                                                       
            out = {n: p for n, p in out.items()}
        return out

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}
        for n, p in self.params.items():
            if self.dataloader is not None:
                p_prev = getattr(self.model, f'{n}_SI_prev_task')
                W = getattr(self.model, f'{n}_W')
                p_cur = p.detach().clone()
                delta = p_cur - p_prev
                # 这部分相比SI delta**2 增加了  0. 5* ewc的权重
                omega_add = W / (1.0 / 2.0 * self._precision_matrices[n] * delta ** 2 + self.epsilon)
                try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                except AttributeError:
                        omega = p.detach().clone().zero_()
                
                # 这部分和SI不一样 进行软更新
                omega_new = 0.5 * omega + 0.5 * omega_add
                n_omega[n] = omega_new
                n_p_prev[n] = p_cur
                # 保存新参数到模型中
                self.model.register_buffer(f'{n}_SI_prev_task', p_cur)
                self.model.register_buffer(f'{n}_SI_omega', omega_new)
                continue

            n_p_prev[n] = p.detach().clone()
            n_omega[n] = p.detach().clone().zero_()
            self.model.register_buffer(f'{n}_SI_prev_task', p.detach().clone())
        return n_p_prev, n_omega

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]
                # 正则项 最终的权重 = ewc-权重 + SI-权重
                _loss = (omega + self._precision_matrices[n]) * (p - prev_values) ** 2
                loss += _loss.sum()
        return loss
  
    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    #  - \sum_k grad *  \theta (t)'
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer(f'{n}_W', self.W[n])
                self.p_old[n] = p.detach().clone()
        return 



# In[20]:


# RWalk
print("RUN Rwalk")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=rwalk(model=model, dataloader=None, epsilon=0.1, device=device)
lll_lambda=100
rwalk_acc = []
task_bar = tqdm(range(len(train_dataloaders)))
prev_guards = []

for train_indexes in task_bar:
    task_bar.set_description(f"Task  {train_indexes+1:02d}")
    print(f'train_indexes={train_indexes}\n', '--'*25)
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, 
                               evaluate=evaluate,device=device, 
                               test_dataloaders=test_dataloaders[:train_indexes+1])
    prev_guards.append(lll_object._precision_matrices)
    lll_object=rwalk(model=model, dataloader=train_dataloaders[train_indexes], epsilon=0.1, device=device, prev_guards=prev_guards)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rwalk_acc.extend(acc_list)

    
print(rwalk_acc)
print("==================================================================================================")


# ## &#x2728; SCP
# Sliced Cramer Preservation
# 
# 参考论文:   
# [Sliced Cramer Preservation](https://openreview.net/pdf?id=BJge3TNKwH)
# 
# 伪代码：  
# 
# ![scp](./HW14_pic/scp.png)

# In[21]:


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return torch.from_numpy(vec)


class scp(object):
    """
    OPEN REVIEW VERSION:
    https://openreview.net/forum?id=BJge3TNKwH
    """
    def __init__(self, model: nn.Module, dataloader, L: int, device, prev_guards=[None]):
        self.model = model 
        self.dataloader = dataloader
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._state_parameters = {}
        self.L= L
        self.device = device
        self.previous_guards_list = prev_guards
        self._precision_matrices = self._calculate_importance()
        for n, p in self.params.items():
            self._state_parameters[n] = p.clone().detach()
    
    def _calculate_importance(self):
        out = {}
        # 初始化 Fisher (F) 矩阵（全部填充0）并加上之前的 guards
        for n, p in self.params.items():
            out[n] = p.clone().detach().fill_(0)
            for prev_guard in self.previous_guards_list:
                if prev_guard:
                    out[n] += prev_guard[n]

        self.model.eval()
        if self.dataloader is not None:
            number_data = len(self.dataloader)                
            for x, y in self.dataloader:
                self.model.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                
                mean_vec = pred.mean(dim=0)
                L_vectors = sample_spherical(self.L, pred.shape[-1])
                L_vectors = L_vectors.transpose(1, 0).to(self.device).float()
                
                total_scalar = 0
                for vec in L_vectors:
                    scalar = torch.matmul(vec, mean_vec)
                    total_scalar += scalar
                total_scalar /= L_vectors.shape[0] 
                total_scalar.backward()         

                for n, p in self.model.named_parameters():                                            
                    out[n].data += p.grad ** 2 / number_data

                out = {n: p for n, p in out.items()}
        return out

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._state_parameters[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def update(self, model):
        return 


# In[22]:


# SCP
print("RUN SLICE CRAMER PRESERVATION")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object = scp(model=model, dataloader=None, L=100, device=device)
lll_lambda = 100
scp_acc = []
task_bar = tqdm(range(len(train_dataloaders)))
prev_guards = []

for train_indexes in task_bar:
    task_bar.set_description(f"Task  {train_indexes+1:02d}")
    print(f'train_indexes={train_indexes}\n', '--'*25)
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    prev_guards.append(lll_object._precision_matrices)
    lll_object=scp(model=model, dataloader=train_dataloaders[train_indexes], L=100, device=device, prev_guards=prev_guards)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scp_acc.extend(acc_list)


print(scp_acc)
print("==================================================================================================")


# #  &#x1F4CC; 绘制不同终生学习`Long Life Machine Learning`算法序列任务学习的情况

# In[25]:


import matplotlib.pyplot as plt


def draw_acc(acc_list, label_list):
    plt.figure(figsize=(16, 4))
    for idx, (acc, label) in enumerate(zip(acc_list, label_list)):
        plt.plot(acc, marker='o', linestyle='--', linewidth=2, markersize=4, label=label)
        
    
    for idx, x in enumerate(range(0, len(acc_list[0]), 10)):
        plt.axvline(x=x, linestyle='--')
        plt.text(x=x+5, y=100, s=f"TASK-{idx+1}", ha='center')
    plt.savefig('acc_summary.png')
    plt.title('acc_summary')
    plt.ylim(60, 102)
    plt.ylabel('ACC')
    plt.legend()
    plt.show() 


acc_list = [baseline_acc, ewc_acc, mas_acc, si_acc, rwalk_acc, scp_acc]
label_list = ['baseline', 'EWC', 'MAS', 'SI', 'RWALK', 'SCP']
# acc_list = [  mas_acc, si_acc, rwalk_acc, scp_acc]
# label_list = [  'MAS', 'SI', 'RWALK', 'SCP']
draw_acc(acc_list, label_list)


# In[ ]:




