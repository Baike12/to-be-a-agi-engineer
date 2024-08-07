import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

# 设置全局的随机种子
def all_seed(seed = 6666):
    """
    设置随机种子
    """
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python 全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')

all_seed(2022)
data_dir = r"D:\ML\data\MLHW_6\faces"

class HeadDataSet(Dataset):
    def __init__(self, fnames, transform):
        super(HeadDataSet, self).__init__()
        self.fnames = fnames
        self.transform = transform
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):# 加载每一个文件时才转换
        fname = self.fnames[idx]
        image = torchvision.io.read_image(fname)
        image = self.transform(image)
        return image

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fname = glob.glob(os.path.join(root,"*"))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),  std=(0.5,0.5,0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = HeadDataSet(fname, transform)
    return dataset

def show_img():
    temp_dataset = get_dataset(os.path.join(data_dir, 'faces'))

    images = [temp_dataset[i] for i in range(16)]# 遍历要看的文件
    grid_img = torchvision.utils.make_grid(images, nrow=4)# 画图，4个窗格
    plt.figure(figsize=(10,10))# 每个窗格大小
    plt.imshow(grid_img.permute(1, 2, 0))# 显示图像
    plt.show()

class Generator(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Generator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, feature_dim*128, bias=False),
            nn.BatchNorm1d(feature_dim*128),
            nn.ReLU()
        )

        self.transposeconv_layer = nn.Sequential(
            self.dconv_bn_relu(feature_dim*8, feature_dim*4),
            self.dconv_bn_relu(feature_dim*4, feature_dim*2),
            self.dconv_bn_relu(feature_dim*2, feature_dim)
        )

        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(
                feature_dim, 3,kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            )
        )


    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.input_layer(x)
        y = y.view(y.size(0), -1, 4, 4)# 一开始生成的图像是4×4的
        y = self.transposeconv_layer(y)
        y = self.out_layer(y)
        return y

def weights_init(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    # 输入（batch， 3， 64， 64）
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim*2),
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               # output -> (batch, 256, 32, 32)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               # output -> (batch, 512, 32, 32)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),  # output -> (batch, 1, 1, 1)
        )
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2,1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)# y = x if x > 0; = ax if x <= 0可以在x < 0的时候保留一定的梯度，防止梯度消失
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)# 变成一维向量
        return y

config = {
    "model_type": "GAN",
    "batch_size": 64,
    "lr": 1e-4,
    "n_epoch": 5,
    "n_critic": 1,
    "z_dim": 100,
    "workspace_dir": data_dir,
}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TrainerGAN():
    def __init__(self, config):
        self.config = config

        self.Gen = Generator(100)
        self.Dis = Discriminator(3)

        self.loss = nn.BCEWithLogitsLoss()

        self.Gen_Optimizer = torch.optim.Adam(self.Gen.parameters(), lr=config['lr'], betas=(0.5, 0.999))# 梯度的移动平均值的一阶矩估计和二阶矩估计
        self.Dis_Optimizer = torch.optim.Adam(self.Dis.parameters(), lr=config['lr'], betas=(0.5, 0.999))# 梯度的移动平均值的一阶矩估计和二阶矩估计

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).to(device)

    def prepare_environment(self):
        """
        训练前环境、数据与模型准备
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 基于时间更新日志和ckpt文件名
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time+f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time+f'_{self.config["model_type"]}')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(self.ckpt_dir)

        # 数据准备：创建dataloader
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

        self.Gen = self.Gen.to(device)
        self.Dis = self.Dis.to(device)
        self.Gen.train()
        self.Dis.train()

    def train(self):
        self.prepare_environment()

        for e, epoch in enumerate(range(self.config["n_epoch"])):
            propress_bar = tqdm(self.dataloader)
            propress_bar.set_description(f"Epoch {e+1}")

            for i, data in enumerate(propress_bar):
                images = data.to(device)
                batch_size = images.size(0)

                z = Variable(torch.rand(batch_size, self.config["z_dim"])).to(device)
                real_images = Variable(images).to(device)
                fake_images = self.Gen(z)
                real_label = torch.ones((batch_size)).to(device)
                fake_label = torch.zeros((batch_size)).to(device)

                real_output = self.Dis(real_images)
                fake_output = self.Dis(fake_images)

                real_loss = self.loss(real_output, real_label)
                fake_loss = self.loss(fake_output, fake_label)
                Dis_loss = (real_loss + fake_loss) / 2

                self.Dis.zero_grad()
                Dis_loss.backward()
                self.Dis_Optimizer.step()

                # train generator
                if self.steps % self.config["n_critic"] == 0:
                    z = Variable(torch.randn(batch_size, self.config["z_dim"])).to(device)
                    fake_images = self.Gen(z)

                    fake_output = self.Dis(fake_images)# 生成器生成的图片有多像真是图片
                    Gen_loss = self.loss(fake_output, real_label)

                    # Generator反向传播
                    self.Gen.zero_grad()
                    Gen_loss.backward()
                    self.Gen_Optimizer.step()

                if self.steps % 10 == 0:
                    propress_bar.set_postfix(loss_G=Gen_loss.item(), loss_D=Dis_loss.item())
                self.steps += 1

            self.Gen.eval()
            f_imgs_sample = (self.Gen(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # 在训练过程中显示图片
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.Gen.train()

            if (e+1) % 5 == 0 or e == 0:
                # 保存checkpoints.
                torch.save(self.Gen.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.Dis.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))

        logging.info('Finish training')
    def inference(self, G_path, n_generate=1000, n_output=30, show=False):
        """
        1. G_path： 生成器ckpt路径
        2. 可以使用此函数生成最终答案
        """

        self.Gen.load_state_dict(torch.load(G_path))
        self.Gen.to(device)
        self.Gen.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).to(device)
        imgs = (self.Gen(z).data + 1) / 2.0

        os.makedirs(os.path.join(data_dir, 'output'), exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], f'output/{i+1}.jpg')

        if show:
            row, col = n_output//10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

def main():
    trainer = TrainerGAN(config)
    trainer.train()

if __name__ == '__main__':
    main()







