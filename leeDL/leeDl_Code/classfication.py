import os.path
import random

import wget
from tqdm import tqdm
import torch

url = "https://github.com/xraychen/shiny-robot/releases/download/v1.0/libriphone.zip"

data_dir = '../data'
data_file = os.path.join(data_dir, 'libriphone.zip')
if not os.path.exists(data_file):
    wget.download(url, out=data_file)

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n,1)
        left = x[n:]
    else:
        return x
    return torch.cat((left, right), dim=0)


def concat_feat1(x, concat_n):
    assert concat_n % 2 == 1 # n 必须是奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    print("origin x:",x.shape)
    x = x.repeat(1, concat_n)
    print("after repeat:",x.shape)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    print("after permute:",x.shape)
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)
    x = x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)
    print("finish concat",x.shape)
    return x


def test_concatn():
    a = torch.arange(1,91)
    a = a.reshape([3,6,5])
    print(a[1])
    print(a[1].shape)
    concat_feat1(a[1], 5)
    print(a[1])
    print(a[1].shape)

def concat_feat(x, concat_n):
    assert concat_n % 2 ==1
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1,0,2)
    mid = concat_n // 2
    for idx in range(1,mid+1):
        x[mid+idx, :] = shift(x[mid+idx], idx)
        x[mid-idx, :] = shift(x[mid-idx], -idx)
    x = x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)
    return x

def preprocess_data(split, phone_flie,feat_dir,concat_nframes, train_val_seed=1337, train_ratio=0.8):
    class_num = 41
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    lable_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_flie, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            lable_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        usage_list = open(os.path.join(phone_flie, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        if split == 'train':
            usage_list = usage_list[:percent]
        else:
            usage_list = usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_flie, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalide split argument for dataset')

    usage_list = [line.strip('\n') for line in usage_list]
    print('number of phone class:', str(class_num), 'length of usage list', str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39*concat_nframes)
    if split != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        print(fname)
        print("feat shape after load:",feat.shape)
        feat = concat_feat1(feat, concat_nframes)
        print("feat shape after concat:",feat.shape)
        if mode != 'test':
            label = torch.LongTensor(lable_dict[fname])
        print("idx:",idx, "cur_len",cur_len)
        X[idx:idx+cur_len, :] = feat
        print("X shape:",feat.shape)
        if mode != 'test':
            y[idx:idx+cur_len] = label

        idx += cur_len

    X= X[:idx,:]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    # print(X.shape)
    if mode != 'test':
        # print(y.shape)
        return X,y
    else:
        return X

from torch.utils.data import Dataset, DataLoader

class LibraDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # print("classifier x shape", x.shape)
        x = self.fc(x)
        return x

# data prarameters
# 用于数据处理时的参数
concat_nframes = 7              # 要连接的帧数,n必须为奇数（总共2k+1=n帧
train_ratio = 0.8               # 用于训练的数据比率，其余数据将用于验证
# training parameters
# 训练过程中的参数
seed = 0                        # 随机种子
batch_size = 512                # 批次数目
num_epoch = 100                   # 训练epoch数
learning_rate = 0.005          # 学习率
model_path = './models/classifier_model.ckpt'     # 选择保存检查点的路径（即下文调用保存模型函数的保存位置）
# model parameters
# 模型参数
input_dim = 39 * concat_nframes # 模型的输入维度，不应更改该值，这个值由上面的拼接函数决定
hidden_layers = 5               # hidden_layer层的数量
hidden_dim = 256                #


import gc
train_X, train_y = preprocess_data('train',phone_flie=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone',feat_dir=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone\feat', concat_nframes=concat_nframes,train_ratio=train_ratio)
valid_X, valid_y = preprocess_data('val',phone_flie=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone',feat_dir=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone\feat', concat_nframes=concat_nframes,train_ratio=train_ratio)

train_set = LibraDataset(train_X, train_y)
valid_set = LibraDataset(valid_X, valid_y)

del train_X, train_y, valid_X, valid_y
gc.collect()

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

import numpy as np

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

same_seeds(seed)
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# def trainer():
best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    val_acc = 0.0
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features, labels = features.to(device),  labels.to(device)

        optimizer.zero_grad()
        output = model(features)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(output, 1)
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss

    if len(valid_set) > 0 :
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader)):
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                output = model(features)

                loss = criterion(output, labels)

                _, valid_pred = torch.max(output,1)
                val_acc = (valid_pred.cpu()==labels.cpu()).sum().item()
                val_loss+= loss

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(valid_set), val_loss/len(valid_loader)
            ))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('save model with acc {:.3f}'.format(best_acc/len(valid_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

if len(valid_set) == 0:
    torch.save(model.state_dict(), model_path)
    print("Done!!")

test_X = preprocess_data('test',phone_flie=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone',feat_dir=r'D:\Note\to-be-a-agi-engineer\Code\data\libriphone\feat', concat_nframes=concat_nframes,train_ratio=train_ratio)
test_set = LibraDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def tester():
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    test_acc = 0.0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)
            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open('./classfication_prediction.csv', 'w') as f:
        f.write('id, class\n')
        for i,y in enumerate(pred):
            f.write('{},{}\n'.format(i,y))



#
# def main():
#     # print('aa')
#     # load_data()
#     # test_concatn()
#     # tester()
#     # trainer()
#
# if __name__ == "__main__":
#     main()
#
#
#



