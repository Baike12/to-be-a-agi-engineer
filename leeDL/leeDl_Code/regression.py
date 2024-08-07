import gdown
import os
import time
import math
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

data_dir = '../data'

if (os.path.exists(os.path.join(data_dir, 'covid.train.csv'))==False):
    gdown.download('https://drive.google.com/uc?id=1kLSW_-cW2Huj7bh84YTdimGBOJaODiOS', os.path.join(data_dir, 'covid.train.csv'), quiet=False)

if (os.path.exists(os.path.join(data_dir, 'covid.test.csv'))==False):
    gdown.download('https://drive.google.com/uc?id=1iiI5qROrAhZn-o4FPqsE97bMzDEFvIdg', os.path.join(data_dir, 'covid.test.csv'), quiet=False)

# print(torch.cuda.is_available())

# 设置随机种子
"""
后端设置为确定性模式
设置随机种子
    numpy的种子
    pytorch的种子
    如果gpu可用，设置gpu的随机种子
"""
def same_seed(seed):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f'Set Seed = {seed}')

"""
数据集拆分:使用random_split将数据拆成训练集和数据集
"""
def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(len(data_set)*valid_ratio)
    train_set_size = len(data_set)-valid_set_size
    train_set, valid_set = random_split(data_set,[train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

"""
预测
    在评估模式下进行
    遍历测试数据中的批量，为了直观看到进度，将测试数据放到tqdm中
    计算模型预测值
    将模型预测值从计算图中分离
    添加到预测结果中
    最后将测试结果沿批量维度加起来
"""
def predict(test_data, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_data):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds,dim=0).numpy()
    return preds

"""
定义数据集：
    标签是空就是测试集
"""
class COVID19DataSet(Dataset):
    """
    定义数据集
    """
    def __init__(self,x, y=None):
        if y is None:
            self.y=y
        else:
            self.y=torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    """
    获取一个元素    
    """
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    """
    返回长度    
    """
    def __len__(self):
        return len(self.x)

"""
网络模型data_file
"""
class My_MOdel(nn.Module):
    """
    三个线性层加上两个relu
    """
    def __init__(self, input_dim):
        super(My_MOdel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
        )
    """
    前向传播
        删除第1维数据，相当于输出一个8维向量
    """
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

"""
特征选择
    提取标签
    提取特征数据
"""
def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1],test_data
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4]
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx],y_train, y_valid


"""
训练器
"""
def trainer(train_loader, valid_loader, model, config, device):
    losser = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=0.9)
    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        train_pbar.set_description(f"Epoch: {epoch+1}/{n_epochs}")

        for x,y in train_pbar:
            optimizer.zero_grad()
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = losser(pred,y)
            loss.backward()
            optimizer.step()
            step+=1
            loss_record.append(loss.detach().item())
            train_pbar.set_postfix({'loss':loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = losser(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'epoch [{epoch+1}/{n_epochs}, Train_loss:{mean_train_loss:.4f}, Valid_loss:{mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid',mean_valid_loss,step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss{:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count +=1

        if early_stop_count >= config['early_stop']:
            print('\n Model is not improving, so we halt the trainning session')
            return


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('***device***', device)
config = {
    'seed': 5201314,      # 随机种子，可以自己填写. :)
    'select_all': True,   # 是否选择全部的特征
    'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'n_epochs': 3000,     # 数据遍历训练次数
    'batch_size': 256,
    'lr': 1e-5,
    'early_stop': 400,    # 如果early_stop轮损失没有下降就停止训练.
    'save_path': './models/model.ckpt'  # 模型存储的位置
}

"""
导入数据
"""
same_seed(config['seed'])
pd.set_option('display.max_columns', 200)
train_df, test_df = pd.read_csv('../data/covid.train.csv'), pd.read_csv('../data/covid.test.csv')
print(train_df.head(3))
train_data, test_data = train_df.values, test_df.values
del train_df, test_df
train_data, valide_data = train_valid_split(train_data, config['valid_ratio'],config['seed'])

print(f'train data size{train_data.shape}, valid data size{valide_data.shape}')

x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valide_data, test_data, config['select_all'])
print(f'numbers of features {x_train.shape[1]}')

train_dataset, valide_dataset, test_dataset = COVID19DataSet(x_train, y_train), COVID19DataSet(x_valid, y_valid), COVID19DataSet(x_test)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valide_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

def train():
    model = My_MOdel(input_dim=x_train.shape[1]).to(device)
    trainer(train_loader,valid_loader,model, config, device)

def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id','tested_positive'])
        for i,p in enumerate(preds):
            writer.writerow([i,p])

def test():
    model = My_MOdel(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)
    save_pred(preds, 'pred.csv')

# train()
test()

