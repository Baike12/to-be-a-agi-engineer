```python
import math
import os.path
import random
import numpy as np
import wget
from tqdm import tqdm
import torch
torch.autograd.set_detect_anomaly(True)

url = "https://www.dropbox.com/s/6l2vcvxl54b0b6w/food11.zip?dl=1"

data_dir = r'D:\ML\data'
data_file = os.path.join(data_dir, 'food11.zip')
if not os.path.exists(data_file):
    wget.download(url, out=data_file)

from torchviz import make_dot

def model_plot(model_class, input_sample):
    """
    打印模型计算图
    """
    clf = model_class()
    y = clf(input_sample)
    clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [('x', input_sample)])) # 将名字参数对转换成列表，再转换成字典，最后加上输入的样本
    return clf_view

# 设置随机种子
def all_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt
from PIL import Image
def quick_observe(train_dir_root):
    pics_path = [os.path.join(train_dir_root, i) for i in os.listdir(train_dir_root)]
    labels = [i.split('_')[0] for i in os.listdir(train_dir_root)]
    idxs = np.arange(len(labels))
    sample_idx = np.random.choice(idxs, size=9, replace=False)
    fig, axes = plt.subplots(3,3,figsize=(20,20))
    for idx_, i in enumerate(sample_idx):
        row = idx_ // 3
        col = idx_ % 3
        image = Image.open(pics_path[i])
        axes[row,col].imshow(image)
        c = labels[i]
        axes[row, col].set_title(f'class_{c}')
    plt.show()

train_dir_root = r'D:\ML\data\food11\training'
# quick_observe(train_dir_root)

import torchvision.transforms as transforms
# 测试集：将图片转换成指定大小，然后转换成张量
test_tfm = transforms.Compose([# 使用Compose构建图像处理流水线
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),# 使用针对IMAGNET的策略进行增强
    transforms.ToTensor(),
])

from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
# 定义数据集
# 初始化：文件路径，转换策略（默认为测试的转换），文件列表（可以为None）
# 长度
# 获取某个下标元素
class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
       super(FoodDataset, self).__init__()
       self.path = path
       self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])# 文件列表就是列出图片的绝对路径，然后排序
       # print(len(self.files))
       self.tranformer = tfm
       if files != None:
          self.files = files
       print(f"one of {path} file is:", self.files[0])

    def __len__(self):
        return len(self.files)

    # 打开一张图片，并进行转换；获取标签：就是文件名中_之前的部分，最后返回图像和标签
    def __getitem__(self, idx):
        fname = self.files[idx]
        # print(fname)
        image = Image.open(fname)
        image = self.tranformer(image)
        try:
            lable = int(fname.split("\\")[-1].split("_")[0])
            # print(lable)
        except:
            lable = -1
        return image, lable

from torchvision.models import resnet50
restNet50 = resnet50(weights=None)
import torch.nn as nn
# 定义分类器模型，包含卷积层和全连接层
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            # 二维卷积、批量归一化、激活函数、最大池化
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),


            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0)
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

# x = torch.randn(1,3,128,128).requires_grad_(True)
# model=model_plot(Classifier, x)
# model.render("cnn", format="png")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
config = {
    'seed': 6666,
    'dataset_dir': r"D:\ML\data\food11",
    'n_epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.0003,
    'weight_decay':1e-5,
    'early_stop': 300,
    'clip_flag': True,
    'save_path': 'D:\ML\models\leeDl_cnn_model.ckpt',
    'resnet_save_path': r'D:\ML\models\leeDl_cnn_model.ckpt'
}
print(device)
all_seed(config['seed'])

def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    save_path = config['save_path']

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf,0,0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_acc= []
        train_pbar = tqdm(train_loader, leave=True, position=0)

        for x,y in train_pbar:
            optimizer.zero_grad()# 由优化器将梯度置0
            X,y = x.to(device), y.to(device)
            # print(X.shape)
            # print(y.shape)
            # print(y)
            output = model(X)
            l = criterion(output, y)
            l.backward()
            # 在梯度下降之前进行梯度裁剪
            if config['clip_flag']:
                grad_norm = nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()

            step+=1
            # 计算精度：输出是一个一维张量，获取其中最大值所在位置
            acc = (output.argmax(dim=-1)==y.to(device)).float().mean()
            train_acc.append(acc.detach().item())
            # 将损失从计算图分离，然后转换成numpy类型
            l_ = l.detach().item()
            loss_record.append(l_)
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})

        # 计算训练精度均值和损失均值：精度或者损失除以样本数量
        mean_train_acc = sum(train_acc) / len(train_acc)
        mean_train_loss = sum(loss_record) / len(loss_record)

        # 验证：不用计算梯度
        model.eval()
        loss_record = []
        valid_acc = []

        for x, y in valid_loader:
            X,y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(X)
                l = criterion(pred, y)
                acc = (pred.argmax(dim=-1)==y.to(device)).float().mean()
            loss_record.append(l.detach().item())
            valid_acc.append(acc.detach().item())

        mean_valid_acc = sum(valid_acc)/len(acc)
        mean_valid_loss = sum(loss_record)/len(loss_record)
        # 打印当前回合、训练精度、训练损失、验证精度、验证损失
        print(f'Epoch {epoch+1}/{n_epochs}, Train acc:{mean_train_acc}, Train loss:{mean_train_loss}, Valid acc:{mean_valid_acc}, Valid loss:{mean_valid_loss}')
        # 如果训练损失<最小最好损失，就停下来，记录模型，并把早停计数置0，否则把早停计数加一，当早停计数>
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), save_path)
            print('Save model with Valid loss:{:.5f}'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count+=1
        if early_stop_count >= config['early_stop']:
            print("\nModle can not improve, so stop train")
            return

# 数据加载
data_dir = config['dataset_dir']
# 先用类封装成数据集，再封装成加载器
train_set = FoodDataset(os.path.join(data_dir, "training"),tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,pin_memory=True)# 将数据放到锁页内存，不会被放到磁盘，加快移动到gpu的速度

valid_set = FoodDataset(os.path.join(data_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'],shuffle=False, pin_memory=True)

model = Classifier().to(device)
trainer(train_loader, valid_loader, model,config, device)
```
### 神经元与感受野
- 一个或者一组神经元守备一个感受野
#### 参数共享
- 当卷积核在扫描的时候，对于每一个感受野，卷积核的参数是不一样的，一部分神经元的参数相同就是参数共享
#### 滤波器
- 当卷积核扫描整张图像，得到特征图，特征图可以看出扫描的图像的卷积核大小的特征
- 一个滤波器扫描一遍图像，得到一个特征图
- 滤波器也会有偏置
#### 特征映射
- 图像通过一个卷积层的一组滤波器，得到一个特征映射
- 特征映射由特征图组成
- 特征映射可以看成一个图像，有滤波器数个通道
#### 下采样
- 拿掉图像中一部分行和列，不影响图像特征
####  汇聚
- 把特征图分成小方块，取其中的最大值或者平均值
- 汇聚的目的是减少计算量，如果算力足够，没必要汇聚

### 归一化
#### 批量归一化
- 对每一层中小批量数据的一个通道进行归一化
- 适合用在卷积层、大批量训练
##### 例子
- 有一个64通道，128×128的特征图，一个批量有32个特征
- 计算这32×128×128个数值的平均值$\mu$ 和方差$\alpha ^{2}$ 
- 对每一个数值$\frac{{x_{i}-\mu}}{\alpha}$ 

#### 层归一化
- 在每个样本的所有特征上进行归一化
- 适合用在RNN，小批量
##### 例子
- 有一个64通道，128×128的特征图，一个批量有32个特征
- 计算这64×128×128个数值的平均值$\mu$ 和方差$\alpha ^{2}$ 
- 对每一个数值$\frac{{x_{i}-\mu}}{\alpha}$ 


#### 软投票
- 多个模型的输出值做加权和
##### 例子
- 三个类别A,B,C，三个模型预测分别为M1(0.1,0.8,0.1)，M2（0.2,0.6,0.2），M3（0.3，0.3,0.4）
- 则等权重下A的软投票为$\frac{{0.1+0.2+0.3}}{3}$ 