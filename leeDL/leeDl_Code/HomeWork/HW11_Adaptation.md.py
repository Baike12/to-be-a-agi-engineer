#!/usr/bin/env python
# coding: utf-8

# #  &#x1F4D1; **作业11： 迁移学习 (领域对抗性训练`Domain Adversarial Training`)**
# 
# > Author: Arvin Liu (r09922071@ntu.edu.tw)
# 
# 如果有任何问题, 联系 [mlta-2022-spring@googlegroups.com](mlta-2022-spring@googlegroups.com)

# # Readme
# 
# 在作业11中，你需要在迁移学习中进行领域对抗性训练。如图左下角所示。
# 
# <img src="./HW11_pic/迁移学习中进行领域对抗性训练.jpg" width="500px">
# 
# ## 场景和为什么领域对抗性训练
# 现在我们有打好标签的源数据以及未被标记目标的数据，其中源数据可能与目标数据相关。我们现在只想用源数据训练模型，并在目标数据上进行测试。
# 
# 如果我们这样做，可能会出现什么问题？ 在我们学习了异常检测之后，我们现在知道，如果我们用源数据中从未出现过的异常数据来测试模型，我们训练的模型很可能会<font color=darkred>导致较差的性能</font>，因为它不熟悉异常数据。
# 
# 例如，我们有一个包含特征提取器(`Feature Extractor`)和分类器(`Classifier`)的模型：
# <img src="./HW11_pic/特征提取器和分类器.jpg" width="500px">
# 
# 模型使用源数据训练，特征提取器(`feature extractor`)会提取有意义的特征，因为它熟悉其分布。从下图中可以看出，蓝点是源数据的分布，已经被聚类为不同的聚类。因此，分类器可以基于这些聚类来预测标签。
# 然而, 在测试目标数据的时候， 特征提取器(`feature extractor`)将不能提取出和源数据相似分布的有意义特征，因为目标特征分布和源数据不一致。这导致为源数据学习出的分类器将不能应用于目标数据。  
# 
# ## 神经网络的领域对抗训练（`Domain Adversarial Training of Nerural Networks (DaNN)`)
# 
# 基于上述问题, DaNN方法在源（训练时间）域和目标（测试时间）域之间建立映射，因此，当与域之间的学习映射相结合时，用源数据学习的分类器也可以应用于目标数据。  
# 
# <img src="./HW11_pic/DaNN.jpg" width="500px">
# 
# 在DaNN中, 作者增加了一个域分类器(`Domain Classifier`),  它是训练框架中经过深度判别训练的分类器(`discriminatively-trained classifeir`)，通过特征提取器提取的特征来区分来自不同领域的数据。随着训练的进行，该方法促进了区分源域和目标域的域分类器，以及特征提取器可以对源数据上主要学习任务具有判别力的特征进行提取，同时对域之间的偏移不加区分的特征进行提取。
# 
# 特征提取器可能优于域分类器，因为域分类器输入是由特征提取器生成的，并且域分类和标签分类的任务不冲突。
# 
# 这种方法导致特征具有区域不变性和相同的特征分布。

# # 数据介绍
# 
# 源数据(`source data`): 真实照片, 
# 目标数据(`target data`): 手绘涂鸦
# 
# 用真实照片和label进行模型训练, 然后尝试对 手绘涂鸦 进行识别类型.
# 
# 数据下载地址： [here](https://github.com/redxouls/ml2020spring-hw11-dataset/releases/download/v1.0.0/real_or_drawing.zip). 
# 下面的代码是数据下载和简单可视化   
# 提示: **源数据和目标数据都是平衡数据，您可以利用这些信息.**

# In[ ]:


# Download dataset
get_ipython().system('wget "https://github.com/redxouls/ml2020spring-hw11-dataset/releases/download/v1.0.0/real_or_drawing.zip" -O real_or_drawing.zip')

# Download from mirrored dataset link
# !wget "https://github.com/redxouls/ml2020spring-hw11-dataset/releases/download/v1.0.1/real_or_drawing.zip" -O real_or_drawing.zip
# !wget "https://github.com/redxouls/ml2020spring-hw11-dataset/releases/download/v1.0.2/real_or_drawing.zip" -O real_or_drawing.zip

# Unzip the files
get_ipython().system('unzip real_or_drawing.zip')


# In[2]:


import matplotlib.pyplot as plt

def no_axis_show(img, title='', cmap=None):
    fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


# In[3]:


titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'tetevision', 'dog', 'dolphin', 'spider']
plt.figure(figsize=(18, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    fig = no_axis_show(plt.imread(f'real_or_drawing/train_data/{i}/{500*i}.bmp'), title=titles[i])

for i in range(10):
    plt.subplot(2, 10, 10+i+1)
    fig = no_axis_show(plt.imread(f'real_or_drawing/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))

plt.show()


# # 特殊领域知识
# 
# 涂鸦时，通常只画出轮廓，因此可以对源数据进行边缘检测处理，使其更接近目标数据。
# 
# 
# ## Canny边缘提取
# Canny边缘检测的实现方法如下。
# 算法的具体原理大家可以自行查阅，可以参考wilki或者是这篇博客 [Canny edge detector 實作](https://medium.com/@pomelyu5199/canny-edge-detector-%E5%AF%A6%E4%BD%9C-opencv-f7d1a0a57d19).
# 
# 用CV2实现Canny边缘检测只需要两个参数：`low_threshold` and `high_threshold`.  
# 示例代码如下： 
# ```python
# cv2.Canny(image, low_threshold, high_threshold)
# ```
# 简单地说，当边值超过`high_threshold`阈值时，我们将其确定为边。如果边值**仅**高于`low_threshold`阈值，我们将确定它是否为边缘。  
# 让我们在源数据上实现它。

# In[4]:


import cv2
import matplotlib.pyplot as plt
titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
plt.figure(figsize=(18, 18))

original_img = plt.imread(f'real_or_drawing/train_data/0/0.bmp')
plt.subplot(1, 5, 1)
no_axis_show(original_img, title='original')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')

canny_50100 = cv2.Canny(gray_img, 50, 100)
plt.subplot(1, 5, 3)
no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

canny_150200 = cv2.Canny(gray_img, 150, 200)
plt.subplot(1, 5, 4)
no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

canny_250300 = cv2.Canny(gray_img, 250, 300)
plt.subplot(1, 5, 5)
no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')
  


# #  &#x2728; 数据处理
#  
# 此数据适用于`torchvision.ImageFolder`。您可以使用`torchvision.ImageFolder`创建数据集。有关图像增强的详细信息，请参阅以下代码中的注释。

# In[5]:


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch import optim 
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


src_transform = transforms.Compose([
    # 将 RGB  转换成 greyscale
    transforms.Grayscale(),
    # cv2 不支持 skimage.Image, 需要先装换成 np.array 并采用 cv2.Canny 算法
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # np.array -> skimag.Image
    transforms.ToPILImage(),
    # 50% 水平翻转(数据增强)
    transforms.RandomHorizontalFlip(),
    # 旋转 +- 15 度,并对旋转后空缺的地方填充0 
    transforms.RandomRotation(15, fill=(0,)),
    # 转成tensor 作为模型输入
    transforms.ToTensor(),
])

# 区别是不进行 Canny 识别轮廓
tgt_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),

])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=src_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=tgt_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)


# #  &#x2728;  模型
# 
# - 特征提取器`Feature Extractor`: 经典的VGG式体系结构
# - 分类器`Label Predictor / Domain Classifier`: 线性模型.

# In[6]:


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x
    

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.layer(x)
        return x
    

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )
        
    def forward(self, h):
        y = self.layer(h)
        return y


# # 预处理
# 
# 优化器： Adam

# In[7]:


feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())


# #  &#x2728; 开始训练
# 
# 
# ## DaNN实施
# 
# 在原始论文中，使用了渐变反转层(` Gradient Reversal Layer`)。特征提取器`Feature Extractor`, 分类器`Label Predictor`, 和域分类器`Domain Classifier` 同时训练.  
# 在我们的代码中，我们先训练域分类器`Domain Classifier`，然后训练我们的特征提取器`Feature Extractor`和`Label Predictor`（与GAN中的生成器和判别器训练过程相同的概念）
# 
# ## 提醒
# * Lambda, 控制领域对抗性损失，在原始论文中是自适应的. 你可以参考论文[Domain-Adversarial Training of Neural Networks](https://arxiv.org/pdf/1505.07818.pdf) . 这里我们设置 `lambda = 0.1`.
# * 目标数据上没有标签, 您只能通过将结果上传到kaggle来评估您的模型. :)

# In[8]:


def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 控制域自适应(domain adaptatoin) 和分类的平衡
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        # 混合 source data 和 target data,否则会误导正在运行的batch_norm的参数(运行源数据和目标数据的 mean/var 不同)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 设置域标签，源数据标签设为1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 训练 domain classifier
        feature = feature_extractor(mixed_data)
        # 在第一部我们不需要训练特征提取器，所以需要对特征进行 .detach() 操作，防止反向传播
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 训练 feature extractor 和 label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = label分类损失 - lamb * 域分类损失.
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')
        
    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

# 训练200 epoch
for epoch in range(200):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.1)

    torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))


# # 预测推理
# 
# 我们使用 pandas 生成csv文件
# 
# 顺便说一句，经过200个时期训练的模型的性能可能不稳定。你可以为更稳定的表现训练更多的时期。

# In[9]:


result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# 生成提交文件
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('DaNN_submission.csv',index=False)


# #  &#x2728; 可视化
# 我们使用t-SNE图来观察提取的特征的分布

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


# ## Step1: 加载检查点并进行评估以获得提取的特征

# In[11]:


# Hints:
# 设置 features_extractor 为 eval 模型
# 开始评估并收集特征和标签
from tqdm import tqdm
feature_extractor.eval()
feat_list = []
label_list = []
for i, (source_data, source_label) in enumerate(tqdm(source_dataloader)):
    source_data = source_data.cuda()
    source_label = source_label.cuda()
    feat = feature_extractor(source_data).cpu().detach().numpy()
    label_list.append(source_label.cpu().detach().numpy())
    feat_list.append(feat)


# In[12]:


labels = np.concatenate(label_list)
labels.shape


# In[13]:


feats = np.concatenate(feat_list)
feats.shape


# ## Step2: 运用 t-SNE 并进行归一化

# In[14]:


# process extracted features with t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(feats)

# Normalization the processed features 
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)


# In[15]:


import seaborn as sns
sne_df = pd.DataFrame(X_norm, columns=['f1', 'f2'])
sne_df['label'] = labels


# In[16]:


sns.scatterplot(data=sne_df, x='f1', y='f2', hue='label')
plt.show()


# # 训练统计
# - 参数量:
#   - Feature Extractor: 2, 142, 336
#   - Label Predictor: 530, 442
#   - Domain Classifier: 1, 055, 233
# 
# - Simple
#  - Training time on colab: ~ 1 hr
# - Medium
#  - Training time on colab: 2 ~ 4 hr
# - Strong
#  - Training time on colab: 5 ~ 6 hrs
# - Boss
#  - **Unmeasurable**

# # 学习率曲线 (Strong Baseline)
# * 这种方法与colab略有不同。
# 
# ![Loss Curve](./HW11_pic/loss.jpg)
# 
# # 精度曲线 (Strong Baseline)
# * 请注意，您无法访问测试准确性。但这张图告诉，即使模型超过了训练数据，测试精度仍在提高，这就是为什么需要训练更多时间的原因。
# 
# ![Acc Curve](./HW11_pic/acc.jpg)
# 

# # Q&A
# 
# 如果有任何与域适配相关的问题，请发送电子邮件至 b08901058@ntu.edu.tw / mlta-2022-spring@googlegroups.com。

# In[ ]:




