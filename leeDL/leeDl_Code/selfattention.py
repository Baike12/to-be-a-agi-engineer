import json
import random

import wget
import os

urls = ["https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partaa",
 "https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partab",
"https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partac",
"https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partad"]

# data_dir = r'D:\ML\data'
# for url in urls:
#     fname = url.split("/")[-1]
#     print(fname)
#     data_file = os.path.join(data_dir, fname)
#     if not os.path.exists(data_file):
#         wget.download(url, out=data_file)

from torchviz import make_dot
# 画图：传入类和输入例子
# 使用make_dot画图
def model_plot(model_class, input_sample):
    mdi = model_class()
    y = mdi(input_sample)
    mli_view = make_dot(y, params=dict(list(mdi.name_parameters())+['x', input_sample]))
    return mli_view

import numpy as np
import torch
import pandas as pd
import os
# 随机种子初始化
# random、cpu、gpu、numpy、pythonhash种子以及后端确定性算法
def all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 如果是cpu只需要手动设置
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # pythonhash
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.enabled=False

all_seed(87)

from pathlib import Path
# 显示数据
def show_file():
    # map中是演讲者与id的键值对
    data_dir = r'D:\ML\data\Dataset'
    map_file = Path(data_dir)/'mapping.json'
    map_json = json.load(map_file.open())
    print(map_json.keys())

    # 每个用户有多个特征文件
    meta_file = Path(data_dir)/'metadata.json'
    meta_json = json.load(meta_file.open())
    print(meta_json.keys())
    # print(meta_json['speakers'].keys())

from torch.utils.data import Dataset
class SpeakerDataset(Dataset):
    def __init__(self, data_dir, segement_len=128):
        super(SpeakerDataset, self).__init__()
        self.data_dir = data_dir
        self.segment_len = segement_len
        # 从mapping中加载演讲者id和演讲者编号的映射
        map_file = Path(data_dir)/'mapping.json'
        map_json = json.load(map_file.open())
        self.speaker2id = map_json['speaker2id']

        # meta中包含了每一个演讲者的多个特征文件名
        meta_file = Path(data_dir)/'metadata.json'
        meta_json = json.load(meta_file.open())['speakers']

        self.speaker_num = len(meta_json.keys())
        self.data = []
        # 构建  特征文件名：演讲者编号 列表，每个列表项是一个包含特征文件名和演讲者序号的小列表
        for speaker, utts in meta_json.items():
            for utt in utts:
                self.data.append([utt['feature_path'], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_path, speaker = self.data[idx]
        mel = torch.load(os.path.join(self.data_dir, feature_path))
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel)-self.segment_len)
            mel = mel[start:start+self.segment_len]
            mel = torch.FloatTensor(mel)
        else:
            mel = torch.FloatTensor(mel)
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
    def get_speaker_number(self):
        return self.speaker_num

class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        test_file = Path(data_dir)/'testdata.json'
        test_json = json.load(test_file.open())
        self.data = test_json['utterances']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt = self.data[idx]
        feature_path = utt['feature+path']
        mel = torch.load(os.path.join(self.data_dir, feature_path))
        return feature_path ,mel

def inference_collate_batch(batch):
    feat_path, mels = zip(*batch)
    return feat_path, torch.stack(mels)


import torch.nn as nn
# 定义模型
class Classifier(nn.Module):
    def __init__(self, input_dim=40, d_model=40,n_spks=600, dropout=0.1):# 输入特征向量维度、transformer特征维度、分类数
        super(Classifier, self).__init__()
        # 将输入维度转换为transformer输入维度
        self.pre_net = nn.Linear(input_dim, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=256,
            nhead=2,
            batch_first=True,
            activation='gelu',
        )
        # 分类层
        self.pred_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_spks)
        )
        
    def forward(self, mels):
        out = self.pre_net(mels)
        out = self.encoder(out)
        # 池化：降维度、防止过拟合、提取更稳定的特征
        stats = out.mean(dim=1)
        return self.pred_net(stats)

import math
from matplotlib import pyplot as plt
# 预热：在前期将学习率慢慢爬升
def plot_lr():
    num_warmup_steps = 1000
    num_training_steps = 70000
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

# plot_lr()
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
# 定义一个预热函数
def get_cosine_schedule_with_warmupo(
        opt:Optimizer,
        num_warmup_steps:int,
        num_train_steps:int,
        num_cycles:float=0.5,# 余弦函数的周期：0.5代表半个周期，也就是衰减到0,1代表一个周期，
        last_epoch:int=-1# 上一个训练周期的epoch数
):
    # 定义一个Lambda函数
    def lambda_lr(current_step):
        if current_step < num_warmup_steps:
            # 在预热阶段：返回当前步在整个预热阶段的进度
            return float(current_step)/float(max(1,num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps)/float(max(1, num_train_steps-num_warmup_steps))
            return max(0.0, 0.5*(1+math.cos(math.pi*float(num_cycles)*2.0*progress)))
    # pytorch定义了一个可以用来进行权重控制的调度器类
    return LambdaLR(opt, lambda_lr, last_epoch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 87,
    'dataset_dir': "D:\ML\data\Dataset",
    'n_epochs': 35,
    'batch_size': 64,

    'scheduler_flag': True,
    'valid_steps': 2000,
    'warmup_steps': 1000,
    # 'total_steps': 70000, # len(train) * n_epochs
    'learning_rate': 1e-3,
    'early_stop': 300,
    'n_workers': 8,
    'save_path': 'D:\ML\models/LeeDl_self_attention_model.ckpt'
}
print(device)
all_seed(config['seed'])

from tqdm import tqdm
def trainer(train_loader, valid_loader, model, config, device, rest_net_flag=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),config['learning_rate'])

    if config['scheduler_flag']:
        scheduler = get_cosine_schedule_with_warmupo(optimizer, config['warmup_steps'], config['n_epochs']*len(train_loader))# 训练步数等于回个乘以训练集大小
    save_path =  config['save_path']

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

from torch.nn.utils.rnn import pad_sequence

counter = 0
def collate_batch(batch):# 合并一个批次的数据
    # 将bath中的mel特征和演讲者提取出来分别变成一个张量
    mel, speaker = zip(*batch)
    # 使用一个很小的数字填充
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    # 转换成浮点型张量并转换成long
    global counter
    counter+=1
    # print(f"speaker{counter}:", speaker)
    return mel, torch.FloatTensor(speaker).long()

from torch.utils.data import random_split, DataLoader

# 获取数据，封装成data_set，转换成Loader
data_dir = config['dataset_dir']
dataset = SpeakerDataset(data_dir)
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


model = Classifier(
    input_dim=40,  # n_mel
    d_model=80,
    n_spks=600,
    dropout=0.1
).to(device)

def main():
    print("train**************")
    trainer(train_loader, valid_loader, model, config, device)

if __name__ == '__main__':
    main()
