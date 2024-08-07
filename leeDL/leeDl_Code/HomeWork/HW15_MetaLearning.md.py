#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **作业 15: Meta Learning**

# # 导入包

# In[1]:


import glob
import random
import os
from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from IPython.display import display
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE = {device}")

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

all_seed(0)


# # 模型构建准备工作
# 
# 由于我们的任务是图像分类，我们需要**建立一个基于CNN的模型**。   
# 然而，要实现MAML算法，**我们需要调整“nn.Module”中的一些代码。**

# MAML伪代码
# 
# \begin{aligned}
#     &\rule{110mm}{0.4pt}                                                                 \\
#     &\text{Algorithm2  MAML for Few-Shot Supervised Learning}\\
#     &\rule{110mm}{0.4pt}                                                                 \\
#     &\textbf{Require: } p(\mathcal{T}): \text{distribution over tasks}\\
#     &\textbf{Require: } \alpha \text{: 一系列task训练-supportSet，梯度更新学习率-在循环内更新} \\ 
#     &\hspace{17mm} \beta \text{: 一系列task评估-querySet，梯度更新学习率-在循环外更新}\\
#     &\rule{110mm}{0.4pt}                                                                 \\
#     &\text{ 1: 初始化参数 } \theta \\
#     &\text{ 2: }\textbf{while }\text{not done }\textbf{do }\\
#     &\text{ 3: }\hspace{5mm}\text{从任务集合中抽取任务 }\mathcal{T}_i \sim  p(\mathcal{T}) \\
#     &\hspace{10mm}\text{这部分和notbook中的 Omniglot、dataloader_init、get_meta_batch 基本一致} \\
#     &\text{ 4: }\hspace{5mm}\textbf{for all }\mathcal{T}_i\textbf{ do }\\
#     &\text{ 5: }\hspace{10mm}\text{从任务中抽取k_shot个样本} \mathcal{D}=\{X^j, Y^j\} \in \mathcal{T}_i\\
#     &\text{ 6: }\hspace{10mm}\text{基于任务的损失函数计算损失} \mathcal{L}_{\mathcal{T}_i}=l(Y^j, f_{\theta_{i}}(X^j))\\
#     &\text{ 7: }\hspace{10mm}\text{基于损失函数计算梯度, 并更新参数} \frac{\partial{\mathcal{L}_{\mathcal{T}_i}}}{\partial \theta_i} = \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta) \\
#     &\hspace{17mm} \theta_i^{\prime} = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta) \\
#     &\text{ 8: }\hspace{10mm}\text{从任务中抽取q_query个样本} \mathcal{D}^{\prime}=\{X^j, Y^j\} \in \mathcal{T}_i\\
#     &\hspace{15mm} \text{基于更新后的}\theta^{\prime}\text{进行预测并计算损失，用于循环后更新} \mathcal{L}^{\prime}_{\mathcal{T}_i}=l(Y^j, f_{\theta^{\prime}_{i}}(X^j))\\
#     &\hspace{15mm} \text{计算梯度}\frac{\partial{\mathcal{L}^{\prime}_{\mathcal{T}_i}}}{\partial \theta^{\prime}_i} = \nabla_\theta \mathcal{L}^{\prime}_{\mathcal{T}_i}(f_{\theta^{\prime}}) \\
#     &\hspace{15mm} \text{计算最终梯度} \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta^{\prime}})  = \frac{\partial{\mathcal{L}^{\prime}_{\mathcal{T}_i}}}{\partial \theta_i}=\frac{\partial{\mathcal{L}^{\prime}_{\mathcal{T}_i}}}{\partial \theta^{\prime}_i}\frac{\partial \theta^{\prime}_i}{\partial \theta_i} \\
#     &\text{ 9: }\hspace{5mm}\textbf{end for}  \\
#     &\text{10: }\hspace{5mm}\text{Update } \theta \leftarrow \theta - \beta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta^{\prime}})  \\
#     &\text{11: }\textbf{end while } \\
#     &\bf{return} \:  \theta                                                     \\[-1.ex]
#     &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
# \end{aligned}
# 
# 
# 在第10行, 我们希望使用初始$\theta$(<font color="#0CC">**模型初始参数**</font> )进行梯度下降（会存在二阶导， 梯度计算可以看第8行），
# 
# 所以在<font color="#0C0">**循环内**</font>（第7行）
# - 我们构建连续梯度图`torch.autograd.grad(loss, fast_weights.values(), create_graph=True)`
# - 并用`functional_forward`进行推理，手动实现SGD进行更新参数，而不是用`forward`和`backward` 
# - 当我们采用`First-order approximation`的时候，直接将连续梯度图关闭就行
#   - 即`torch.autograd.grad(loss, fast_weights.values(), create_graph=False)`
# 

# ## **Step 1: 模型块定义`Model block definition`**

# In[2]:


def ConvBlock(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


# 利用指定权重进行foward
def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(
        x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True
    )
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


# ### 模型定义

# In[3]:


class Classifier(nn.Module):
    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        """
        使用指定参数进行推理
        params:
            x: 输入图片 [batch, 1, 28, 28]
            params: OrderedDict 模型参数,
                i.e. 卷积层的 weights 和 biases 与 `batch normalization`的 weights 和 biases
        """
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(
                x,
                params[f"conv{block}.0.weight"],
                params[f"conv{block}.0.bias"],
                params.get(f"conv{block}.1.weight"),
                params.get(f"conv{block}.1.bias"),
            )
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params["logits.weight"], params["logits.bias"])
        return x


# ## **Step 2: 创建标签**
# 
# `create_label` 用于创建标签
# 
# 对于`N-way K-shot few-shot`分类问题中,
# - `n_way`  表示n个类别, 
# - `k_shot` K表示每个类的样本数.  
# 

# In[4]:


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


create_label(5, 2)


# ## **Step 3: 计算`Accuracy`**

# In[5]:


def calculate_accuracy(logits, labels):
    """utility function for accuracy calculation"""
    acc = np.asarray(
        [(torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy())]
    ).mean()
    return acc


# ## **Step 4: 定义 Dataset**
# 
# `dataset` 返回随机抽取的一个类型的 (`k_shot + q_query`)张图片
# 
# 返回张量的大小为： `[k_shot + q_query, 1, 28, 28]`.  
# 

# In[6]:


data_dir = '../input/ml2022spring-hw15/omniglot'
train_data_path = f"{data_dir}/Omniglot/images_background/"
file_list = [
            f for f in glob.glob(train_data_path + "**/character*", recursive=True)
        ]
len(file_list)


# In[7]:


file_list[:10]


# In[8]:


class Omniglot(Dataset):
    def __init__(self, data_dir, k_shot, q_query, task_num=None):
        # 路径tree如下
        #         ../input/ml2022spring-hw15/omniglot/Omniglot/images_background
        #         ├── Alphabet_of_the_Magi.0
        #         │   ├── character01
        #         │   │   ├── 0709_01.png
        #         │   │   ├── 0709_02.png
        #         │   │   ├── ...
        #         │   ├── character02
        #         │   │   ├── 0710_01.png
        #         │   │   ├── 0710_02.png
        #         │   │   ├── ...
        #         │   ├── character03
        #         │   │   ├── 0711_01.png
        #         │   │   ├── 0711_02.png
        #         │   │   ├── ...
        # 获取所有classifier :  dir/[type]/character[x]
        self.file_list = [
            f for f in glob.glob(data_dir + "**/character*", recursive=True)
        ]
        # 限制 task_num 数量的classifier
        if task_num is not None:
            self.file_list = self.file_list[: min(len(self.file_list), task_num)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        # 输出
        self.n = k_shot + q_query

    def __getitem__(self, idx):
        # 取其中的一个 classifier
        img_path = self.file_list[idx]
        img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
        img_list.sort()
        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        
        # 每个 classifier 随机抽取 `k_shot + q_query` 张img
        sample = np.arange(20)
        np.random.shuffle(sample)
        random_idx_list = sample[:self.n]
        imgs = torch.stack(imgs)[random_idx_list]
        return imgs

    def __len__(self):
        return len(self.file_list)


# ## &#x2728; **Step 5: 算法实现`Learning Algorithms`**
# 
# ### 迁移学习`Transfer learning`
# 
# `BaseSolver`首先会从训练集中抽取5个任务， 然后在5个任务上依次进行正常的分类器训练。  
# 在推理阶段，模型在`support`样本上进行`inner_train_step`微调， 然后在`query`数据上进行推理  
# 为了与元学习(`meta learning`)求解器保持一致，基础求解器具有与元学习解算器完全相同的输入和输出格式。

# In[9]:


def BaseSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=1,
    inner_lr=0.4,
    train=True,
    return_labels=False,
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    for meta_batch in x:
        # 获取数据 
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot :]

        if train:
            """ training loop """
            # 使用support set计算损失
            labels = create_label(n_way, k_shot).to(device)
            logits = model.forward(support_set)
            loss = criterion(logits, labels)

            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, labels))
        else:
            """ validation / testing loop """
            # 用 support set 图片进行 `inner_train_step` 微调
            fast_weights = OrderedDict(model.named_parameters())
            for inner_step in range(inner_train_step):
                train_label = create_label(n_way, k_shot).to(device)
                logits = model.functional_forward(support_set, fast_weights)
                loss = criterion(logits, train_label)

                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                # Perform SGD
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            if not return_labels:
                """ validation """
                val_label = create_label(n_way, q_query).to(device)

                logits = model.functional_forward(query_set, fast_weights)
                loss = criterion(logits, val_label)
                task_loss.append(loss)
                task_acc.append(calculate_accuracy(logits, val_label))
            else:
                """ testing """
                logits = model.functional_forward(query_set, fast_weights)
                labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    batch_loss = torch.stack(task_loss).mean()
    task_acc = np.mean(task_acc)

    if train:
        # 更新model
        model.train()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return batch_loss, task_acc


# ### &#x2728; Meta Learning
# 
# 这里是Meta Learning algorithm的主要实现  
# <font color=darkred><b>TODO: </font></b>
# - <font color=darkred>实现`First Order MAML`, </font>可以参考[p.25 of the slides](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta1%20(v6).pdf#page=25&view=FitW).
# - <font color=darkred>实现一般的`original MAML`, </font>可以参考[the slides of meta learning (p.13 ~ p.18)](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta1%20(v6).pdf#page=13&view=FitW).
# 
# 
# 

# In[10]:


def MetaSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=1,
    inner_lr=0.4,
    train=True,
    return_labels=False,
    FO=False
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    for meta_batch in x:
        # 获取数据
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot :]
        # 没有training loop 
        # 复制原始参数
        fast_weights = OrderedDict(model.named_parameters())
        ### ---------- INNER TRAIN LOOP ---------- ###
        # support_set 进行1step训练： 关注梯度——一阶导
        for inner_step in range(inner_train_step):
            train_label = create_label(n_way, k_shot).to(device)
            logits = model.functional_forward(support_set, fast_weights)
            loss = criterion(logits, train_label)
            """ Inner Loop Update """
            # TODO: 这里实现MAML
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=not FO) # 便于进行二阶导
            fast_weights = OrderedDict(
                (name, param - inner_lr * (grad.detach().data if FO else grad) )
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )

        ### ---------- INNER VALID LOOP ---------- ###
        if not return_labels:
            """ training / validation """
            # query_set 进行测试： 关注loss——二阶导
            val_label = create_label(n_way, q_query).to(device)
            logits = model.functional_forward(query_set, fast_weights)
            loss = criterion(logits, val_label)
            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, val_label))
        else:
            """ testing """
            logits = model.functional_forward(query_set, fast_weights)
            labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()
    task_acc = np.mean(task_acc)
    if train:
        """ Outer Loop Update """
        # TODO: 二阶梯度方向传播
        meta_batch_loss.backward()
        optimizer.step()

    return meta_batch_loss, task_acc


# ## **Step 6: 初始化**
# 
# 模型及数据初始化。

# In[11]:


n_way = 5
k_shot = 1
q_query = 1
train_inner_train_step = 1
val_inner_train_step = 3
inner_lr = 0.4
meta_lr = 0.001
meta_batch_size = 32
max_epoch = 30
eval_batches = 20
data_dir = '../input/ml2022spring-hw15/omniglot'
train_data_path = f"{data_dir}/Omniglot/images_background/"


# ### Dataloader初始化

# In[12]:


def dataloader_init(datasets, shuffle=True, num_workers=2):
    train_set, val_set = datasets
    # 这里batch_size设置成n_way
    # 返回 [n_way, k_shot + q_query, 1, 28, 28]
    train_loader = DataLoader(
        train_set,
        batch_size=n_way,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=n_way, num_workers=num_workers, shuffle=shuffle, drop_last=True
    )

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    return (train_loader, val_loader), (train_iter, val_iter)


# ### Model & optimizer 初始化

# In[16]:


def model_init():
    meta_model = Classifier(1, n_way).to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    loss_fn = nn.CrossEntropyLoss().to(device)
    return meta_model, optimizer, loss_fn


# ### 获取`meta-batch`方法
# 
# 主要的作用是将 `[n_way, k_shot+q_query, 1, 28, 28]` 转变成
# `[n_way*k_shot + n_way*q_query, 1, 28, 28]` 便于在Solver中拆分成 `support_set` 和 `query_set`

# In[17]:


def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    """
    主要的作用是将 [n_way, k_shot+q_query, 1, 28, 28] 转变成
    [n_way*k_shot + n_way*q_query, 1, 28, 28] 便于在Solver中拆分成 support_set 和 query_set
    """
    data = []
    for _ in range(meta_batch_size):
        try:
            # 一个 "task_data" 张量代表 一个task的data: 大小为 [n_way, k_shot+q_query, 1, 28, 28]
            task_data = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            task_data = next(iterator) 
        train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
        val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
        task_data = torch.cat((train_data, val_data), 0)
        data.append(task_data)
    return torch.stack(data).to(device), iterator


# # &#x2728; **训练与测试**

# ## 开始训练
# - `solver = 'base'`: 迁移学习(` transfer learning algorithm.`)
# - `solver = 'meta'`: 元学习(`meta learning algorithm`)
# 

# In[18]:


from functools import partial
meta_lr_org = meta_lr = 0.001
solver = 'meta' # base, meta
FO = False
meta_model, optimizer, loss_fn = model_init()

# 基于solver初始化训练数据
if solver == 'base':
    f_max_epoch = 5 # the base solver 只用 5 epochs
    meta_lr = meta_lr_org
    print(f'use transferLearning & f_max_epoch={f_max_epoch} & meta_lr={meta_lr}')
    Solver = BaseSolver
    train_set, val_set = torch.utils.data.random_split(
        Omniglot(train_data_path, k_shot, q_query, task_num=10), [5, 5]
    )
    (train_loader, val_loader), (train_iter, val_iter) = dataloader_init((train_set, val_set), shuffle=False)

elif solver == 'meta':
    f_max_epoch = max_epoch
    meta_lr = meta_lr_org
    if FO:
        f_max_epoch = 40
        meta_lr = 0.0014
        print(f'use FO-MAML & f_max_epoch={f_max_epoch} & meta_lr={meta_lr}')
    else:
        print(f'use MAML & f_max_epoch={f_max_epoch} & meta_lr={meta_lr}')
    
    Solver = partial(MetaSolver, FO=FO)
    dataset = Omniglot(train_data_path, k_shot, q_query)
    train_split = int(0.8 * len(dataset))
    val_split = len(dataset) - train_split
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_split, val_split]
    )
    (train_loader, val_loader), (train_iter, val_iter) = dataloader_init((train_set, val_set))
else:
    raise NotImplementedError


train_bar = tqdm(range(f_max_epoch))
for epoch in train_bar:
    train_bar.set_description(f"[ Epoch {epoch+1:02d}/{f_max_epoch:02d} ]")
    train_meta_loss = []
    train_acc = []
    # The "step" here is a meta-gradinet update step
    for step in tqdm(range(max(1, len(train_loader) // meta_batch_size))):
        x, train_iter = get_meta_batch(
            meta_batch_size, k_shot, q_query, train_loader, train_iter
        )
        meta_loss, acc = Solver(
            meta_model,
            optimizer,
            x,
            n_way,
            k_shot,
            q_query,
            loss_fn, 
            inner_train_step=train_inner_train_step
        )
        train_meta_loss.append(meta_loss.item())
        train_acc.append(acc)

    print("--"*25, f'{epoch+1:02d}', "--"*25)
    print("  Loss    : ", "%.3f" % (np.mean(train_meta_loss)), end="\t")
    print("  Accuracy: ", "%.3f %%" % (np.mean(train_acc) * 100))

    # 每个epoch训练完后查看验证集的表现(validation accuracy)  
    # 同样也可以在验证集验证的后实现`Early stopping` (可以参考 HW01 中的实现)
    val_acc = []
    val_loss = []
    for eval_step in range(max(1, len(val_loader) // (eval_batches))):
        x, val_iter = get_meta_batch(
            eval_batches, k_shot, q_query, val_loader, val_iter
        )
        # test的时候进行 3次inner steps 更新参数
        val_loss_i, acc = Solver(
            meta_model,
            optimizer,
            x,
            n_way,
            k_shot,
            q_query,
            loss_fn,
            inner_train_step=val_inner_train_step,
            train=False,
        )
        val_acc.append(acc)
        val_loss.append(val_loss_i.item())

    train_bar.set_postfix({
        "trainLoss": "%.3f" % (np.mean(train_meta_loss)),
        "trainAccuracy": "%.3f %%" % (np.mean(train_acc) * 100),
        "valLoss": "%.3f" % (np.mean(val_loss)),
        "valAccuracy": "%.3f %%" % (np.mean(val_acc) * 100)
    })
    print("**"*25)
    print("  Validation accuracy: ", "%.3f %%" % (np.mean(val_acc) * 100))


# ## 测试和结果输出
# 
# 由于测试数据是由TA提前采样的，因此不应更改“OmnigloTest”数据集中的代码，否则在Kaggle排行榜上的分数可能不正确。  
# 
# 但是，可以随意更改变量`inner_train_step`来设置`query`集图像上的训练步骤。

# In[19]:


import os

class OmniglotTest(Dataset):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.n = 5

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        support_files = [
            os.path.join(self.test_dir, "support", f"{idx:>04}", f"image_{i}.png")
            for i in range(self.n)
        ]
        query_files = [
            os.path.join(self.test_dir, "query", f"{idx:>04}", f"image_{i}.png")
            for i in range(self.n)
        ]

        support_imgs = torch.stack(
            [self.transform(Image.open(e)) for e in support_files]
        )
        query_imgs = torch.stack([self.transform(Image.open(e)) for e in query_files])

        return support_imgs, query_imgs

    def __len__(self):
        return len(os.listdir(os.path.join(self.test_dir, "support")))


# In[20]:


test_inner_train_step = 10 # 可以更改这里

test_batches = 20
test_data_path = '../input/ml2022spring-hw15/omniglot-test/Omniglot-test'
test_dataset = OmniglotTest(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=test_batches, shuffle=False)

output = []
for _, batch in enumerate(tqdm(test_loader)):
    support_set, query_set = batch
    x = torch.cat([support_set, query_set], dim=1)
    x = x.to(device)

    labels = Solver(
        meta_model,
        optimizer,
        x,
        n_way,
        k_shot,
        q_query,
        loss_fn,
        inner_train_step=test_inner_train_step,
        train=False,
        return_labels=True,
    )

    output.extend(labels)

# 写入 csv
with open("output.csv", "w") as f:
    f.write(f"id,class\n")
    for i, label in enumerate(output):
        f.write(f"{i},{label}\n")


# # **参考**
# 1. Chelsea Finn, Pieter Abbeel, & Sergey Levine. (2017). [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.](https://arxiv.org/abs/1909.09157)
# 1. Aniruddh Raghu, Maithra Raghu, Samy Bengio, & Oriol Vinyals. (2020). [Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.](https://arxiv.org/abs/1909.09157)

# In[ ]:




