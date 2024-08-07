#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **作业10：对抗攻击(Adversarial Attack)**
# 
# PPT: https://reurl.cc/7DDxnD
# 
# 联系邮箱: ntu-ml-2022spring-ta@googlegroups.com

# ## 环境设设置 & 数据下载
# 
# 我们使用 [pytorchcv](https://pypi.org/project/pytorchcv/) 获取 `CIFAR-10` 预训练模型   
# 所以我们需要先建立环境。我们还需要下载我们想要攻击的数据（200张图像）。

# In[2]:


# 设置环境
get_ipython().system('pip install pytorchcv')
get_ipython().system('pip install imgaug')

# 下载数据
get_ipython().system('wget https://github.com/DanielLin94144/ML-attack-dataset/files/8167812/data.zip')

# 解压
get_ipython().system('unzip ./data.zip')
get_ipython().system('rm ./data.zip')


# In[3]:


import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8


# ## 全局设置
# #### **[NOTE]**: 不要更改此处的设置，否则您生成的图像可能不符合限制
# * $\epsilon$ 设定为固定值 8. 但是在 **Data section**, 我们将首先对原始像素值应用变换 (0-255)范围 **转换成 (0-1)范围** 然后 **标准化 (减去mean除以std)**. 所以$\epsilon$ 在攻击期间应该设置为 $\frac{8}{255 * std}$ 。
# 
# * 解释 (optional)
#     * 将原始图像的第一个像素表示为$p$，将对抗性图像的第一像素表示为$a$。
#     * $\epsilon$ 约束告诉我们 $\left| p-a \right| <= 8$.
#     * ToTensor()函数为 $T(x) = x/255$
#     * Normalize()函数为 $N(x) = (x-mean)/std$， $mean$ 和 $std$ 是常数
#     * 对 $p$ 和 $a$ 进行ToTensor()和 Normalize() 后 , 约束变成 $\left| N(T(p))-N(T(a)) \right| = \left| \frac{\frac{p}{255}-mean}{std}-\frac{\frac{a}{255}-mean}{std} \right| = \frac{1}{255 * std} \left| p-a \right| <= \frac{8}{255 * std}.$
#     * 所以，在经过 ToTensor() 和 Normalize()之后，我们需要设置 $\epsilon$ 为 $\frac{8}{255 * std}$ .

# In[4]:


# 平均值和标准差是根据cifar10数据集计算的统计数据
cifar_10_mean = (0.491, 0.482, 0.447) # cifar_10 图片数据三个通道的均值
cifar_10_std = (0.202, 0.199, 0.201) # cifar_10 图片数据三个通道的标准差

# 将mean和std转换为三维张量，用于未来的运算
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8/255/std


# In[5]:


root = './data' # 用于存储`benign images`的目录
# benign images: 不包含对抗性扰动的图像
# adversarial images: 包括对抗性扰动的图像


# ## Data
# 
# 从根目录构建数据集和数据加载器。请注意，我们存储每个图像的文件名以备后续使用。

# In[6]:


import os
import glob
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')


# ## &#x2728; Utils -- `Benign Images`评估

# In[7]:


# 评估模型在良性图像上的性能
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)


# ## &#x2728; Utils -- 攻击算法(`Attack Algorithm`)
# 

# In[8]:


def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # 用良性图片初始化 x_adv
    x_adv.requires_grad = True # 需要获取 x_adv 的梯度
    loss = loss_fn(model(x_adv), y) # 计算损失
    loss.backward()  
    # fgsm: 在x_adv上使用梯度上升来最大化损失
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv

# 在“全局设置”部分中将alpha设置为步长 
# alpha和num_iter可以自己决定设定成何值
alpha = 0.8 / 255 / std
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    # num_iter 次迭代
    for i in range(num_iter):
        x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # 用（ε=α）调用fgsm以获得新的x_adv
        # x_adv = x_adv.detach().clone()
        # x_adv.requires_grad = True  
        # loss = loss_fn(model(x_adv), y)  
        # loss.backward()  
        # grad = x_adv.grad.detach()
        # x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # x_adv 裁剪到 [x-epsilon, x+epsilon]范围
    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=1.0):
    x_adv = x
    # 初始化 momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # num_iter 次迭代
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True  
        loss = loss_fn(model(x_adv), y)  
        loss.backward()  
        # TODO: Momentum calculation
        grad = x_adv.grad.detach() + (1 - decay) * momentum
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # x_adv 裁剪到 [x-epsilon, x+epsilon]范围
    return x_adv


# ## &#x2728; Utils -- Attack
# * 召回(Recall)
#   * ToTensor()函数为 $T(x) = x/255$
#   * Normalize()函数为 $N(x) = (x-mean)/std$， $mean$ 和 $std$ 是常数
# 
# * 反函数(Inverse function)
#   * 反 Normalize() 函数为  $N^{-1}(x) = x*std+mean$ ，$mean$ 和 $std$ 是常数
#   * 反 ToTensor()  函数为  $T^{-1}(x) = x*255$.
# 
# * **特别注意事项**
#   * ToTensor() 同时也会变换图片的shape `(height, width, channel)` -> `(channel, height, width)`, 所以我们还需要将形状转换回原始形状。
#   * 由于我们的数据加载器对一批数据进行采样，因此我们需要的是转置 **`(batch_size, channel, height, width)`** 变回 **`(batch_size, height, width, channel)`** 使用`np.transpose`.

# In[9]:


# 执行对抗性攻击 并 生成对抗性示例
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn) # 获得对抗性示例
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # 保存对抗性示例
        adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
        adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# 创建存储对抗性示例的目录
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8)) # 图片数据需要转成 uint8
        im.save(os.path.join(adv_dir, name))


# ## Model / Loss Function
# 
# 可用模型列表 [here](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py). 
# 请选择后缀为_cifar10的模型。无法访问/加载某些模型，可以直接跳过，因为TA的模型不会使用这些类型的模型。

# In[10]:


from pytorchcv.model_provider import get_model as ptcv_get_model

model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()

benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'[ Base(未Attack图片评估) ] benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')


# ## FGSM

# In[11]:


adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
print(f'[ Attack(FGSM Attack图片评估) ] fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')

create_dir(root, 'fgsm', adv_examples, adv_names)


# ## I-FGSM

# In[12]:


adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
print(f'[ Attack(I-FGSM Attack图片评估) ] ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')

create_dir(root, 'ifgsm', adv_examples, adv_names)


# ## 压缩打包图像
# * 上传压缩文件(·.tgz·)到 [JudgeBoi： https://ml.ee.ntu.edu.tw/hw10/](https://ml.ee.ntu.edu.tw/hw10/)

# In[13]:


get_ipython().run_line_magic('cd', 'fgsm')
get_ipython().system('tar zcvf ../fgsm.tgz *')
get_ipython().run_line_magic('cd', '..')

get_ipython().run_line_magic('cd', 'ifgsm')
get_ipython().system('tar zcvf ../ifgsm.tgz *')
get_ipython().run_line_magic('cd', '..')


# ## 集合攻击示例
# * 集成多个模型作为代理模型，以提高黑匣子的可转移性 ([paper](https://arxiv.org/abs/1611.02770))
# 
# <font color=darkred><b>***TODO***: 将多模型预测结果（logits）累加 </font></b>
# 

# In[21]:


class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        model
        for i, m in enumerate(self.models):
        # TODO: sum up logits from multiple models  
            if i == 0:
                res = m(x)
                continue
            res += m(x)
        return self.softmax(res)


# In[22]:


model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'preresnet20_cifar10'
]
ensemble_model = ensembleNet(model_names).to(device)
ensemble_model.eval()


# ## Visualization

# In[13]:


import matplotlib.pyplot as plt

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 20))
cnt = 0
for i, cls_name in enumerate(classes):
    path = f'{cls_name}/{cls_name}1.png'
    # 未Attack图片（benign image）
    cnt += 1
    plt.subplot(len(classes), 4, cnt)
    im = Image.open(f'./data/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
    # Attack后图片（adversarial image）
    cnt += 1
    plt.subplot(len(classes), 4, cnt)
    im = Image.open(f'./fgsm/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
plt.tight_layout()
plt.show()


# ## 报告问题
# * 请确保您遵循以下设置：源模型是`resnet110_cifar10`，对“dog2.png”应用vanilla fgsm攻击。您可以在“fgsm/dog2.png'”中找到受干扰的图像。

# In[19]:


# original image
path = f'dog/dog2.png'
im = Image.open(f'./data/{path}')
logit = model(transform(im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'benign: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(np.array(im))
plt.tight_layout()
plt.show()

# adversarial image 
im = Image.open(f'./fgsm/{path}')
logit = model(transform(im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(np.array(im))
plt.tight_layout()
plt.show()


# ## &#x2728; 被动防御(`Passive Defense`)-JPEG压缩
# 通过imgaug包进行JPEG压缩，压缩率设置为70
# 
# 参考: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html#imgaug.augmenters.arithmetic.JpegCompression

# In[20]:


import imgaug.augmenters as iaa

# 预处理image
x = transforms.ToTensor()(im)*255
x = x.permute(1, 2, 0).numpy()
compressed_x = x.astype(np.uint8)


logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'JPEG adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(compressed_x)
plt.tight_layout()
plt.show()


# TODO: use "imgaug" package to perform JPEG compression (compression rate = 70)
cmp_model = iaa.arithmetic.JpegCompression(compression=70)
compressed_x = cmp_model(images=compressed_x)

logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'JPEG assive Defense: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(compressed_x)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




