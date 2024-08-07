## GAN
### 数据集下载与准备
#### 获取并处理数据集
##### 流程
- 
##### 实现
```python
class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples
```
#### 获取数据：把图像转换成张量
##### 流程
- 转换成pil格式
- 调整大小
- 转换成张量
- 归一化
- 组合成一个流水线
##### 实现
```python
def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))# 匹配根目录下所有文件
    compose = [
        transforms.ToPILImage(),# 转换成pil格式图像
        transforms.Resize((64, 64)),# 调整大小
        transforms.ToTensor(),# 转换成张量
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),# 归一化
    ]
    transform = transforms.Compose(compose)# 组合成一个单一转换操作
    dataset = CrypkoDataset(fnames, transform)# 处理数据集
    return dataset
```
#### 加载图像看一下
##### 流程
- 
##### 实现
```python
temp_dataset = get_dataset(os.path.join(workspace_dir, 'faces'))

images = [temp_dataset[i] for i in range(4)]# 遍历要看的文件
grid_img = torchvision.utils.make_grid(images, nrow=4)# 画图，4个窗格
plt.figure(figsize=(10,10))# 每个窗格大小
plt.imshow(grid_img.permute(1, 2, 0))# 显示图像
plt.show()
```
### 生成器 class Generator(nn.Module):

#### 初始化
##### 原理
- 输入一个随机噪声，经过生成器生成一个图像，由判别器来判断是否是预期的图像
- 使用转置卷积层将随机噪声逐渐变成图像
##### 流程
- 输入层
- 转置卷积层
- 输出层
##### 实现
```python
    def __init__(self, in_dim, feature_dim=64):
        super(Generator, self).__init__()
    
        #input: 输入随机一维向量 (batch, 100) 随机生成噪点数据 -> (batch, 64 * 8 * 4 * 4)
        self.l1 = nn.Sequential(# 输入一个随机噪声
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),# 输出一个64×64的张量
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),# 添加批量归一化层
            nn.ReLU()
        )
        # y.view(y.size(0), -1, 4, 4) -> 转成 (batch, feature_dim * 8, 4, 4)
        # 上采样并提取特征：逐步将channel中的特征信息转到 height and width 维度
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),               # out_put -> (batch, feature_dim * 4, 8, 8)     
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),               # out_put -> (batch, feature_dim * 2, 16, 16)     
            self.dconv_bn_relu(feature_dim * 2, feature_dim),                   # out_put -> (batch, feature_dim, 32, 32)     
        )
        # out_put -> (batch, 3, 64, 64) channel dim=1
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()   
        )
        self.apply(weights_init)

```

#### 转置卷积
##### 原理
- 将图像高和宽变成两倍：（输入大小-1）× 步幅 - 2×填充 + 卷积核大小 + 输出填充
	- （64-1）×2 - 2×2 + 5 + 1 = 128
##### 流程
- 融合转置卷积，二维批量归一化、激活函数
##### 实现
```python
    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,# 转置卷积
                               padding=2, output_padding=1, bias=False),        # 双倍 height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
```

#### 前向传播
##### 原理
- 
##### 流程
- 
##### 实现
```python
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)#
        y = self.l2(y)
        y = self.l3(y)
        return y
```
#### 权重初始化
##### 原理
- 批量归一化一般在relu=max（0，x），如果均值初始化为0会导致很多值<0，被截断为0
##### 流程
- 
##### 实现
```python
def weights_init(m):
    classname = m.__class__.__name__# 获取类名
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
```
#### 转置卷积就是卷积的反向操作
##### 原理
- 在通道保持不变的情况下，通过上采样使特征图空间变大
- 转置卷积的输出大小：（输入大小-1）× 步幅 - 2×填充 + 卷积核大小 + 输出填充
##### 流程
- 
##### 实现
```python
def test_transposeconv1():
    batch_size = 1
    feature_dim = 64
    input_tensor = torch.randn(batch_size, feature_dim * 8, 4, 4)

    # 定义转置卷积层
    conv_transpose = nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)

    # 前向传播
    output_tensor = conv_transpose(input_tensor)

    print("Input Tensor Shape: ", input_tensor.shape)
    print("Output Tensor Shape: ", output_tensor.shape)
```

### 判别器
#### 卷积
##### 原理
- 卷积输出大小计算：（输入大小 + 2×填充 - 卷积核大小） / 步幅 + 1
- （32 + 2 × 1 - 4） / 2 + 1 = 16
##### 流程
- 
##### 实现
```python
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), # output -> (batch, 64, 32, 32)

```
#### 整体流程
##### 原理
- 利用卷积来判断
##### 流程
- 
##### 实现
```python
class Discriminator(nn.Module):
    """
    输入: (batch, 3, 64, 64)
    输出: (batch)
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()
            
        # input: (batch, 3, 64, 64)
        """
        设置Discriminator的注意事项:
            在WGAN中需要移除最后一层 sigmoid
        """
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), # output -> (batch, 64, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   # output -> (batch, 128, 32, 32)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               # output -> (batch, 256, 32, 32)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               # output -> (batch, 512, 32, 32)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),  # output -> (batch, 1, 1, 1)
            nn.Sigmoid() 
        )
        self.apply(weights_init)
        
    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        设置Discriminator的注意事项:
            在WGAN-GP中不能使用 nn.Batchnorm， 需要使用 nn.InstanceNorm2d 替代
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y

```

### 训练器
#### BCELoss
##### torch.nn.CrossEntropyLoss
$$\begin{align}
L_{i} &= -\sum_{k=1}^{N} y_{ik}\log(p_{ik})
\end{align}$$
- 第i个样本的交叉熵等于
	- 标签中的实际类别 × 模型输出的实际类别的概率
	- 因为标签中只有一个非0，所以实际上就两个数相乘

##### torch.nn.BCELoss
- 通常用于二分类
$$\begin{align}
L_{i} &= -(y_{ji}\log(p_{i})+(1-y_{i})\log(1-p_{i}))
\end{align}$$
- $y_{i}$ 是第i个样本实际的标签（0或者1），$p_{j}$ 是模型预测为正类的概率
#### 初始化与准备数据
##### 原理
- 
##### 流程
- 初始化
- 准备数据
##### 实现
```python
class TrainerGAN():
    def __init__(self, config):
        self.config = config
        
        self.G = Generator(100)
        self.D = Discriminator(3)
        
        self.loss = nn.BCELoss()

        """
        优化器设置注意：
            GAN: 使用 Adam optimizer
            WGAN: 使用 RMSprop optimizer
            WGAN-GP: 使用 Adam optimizer 
        """
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        
        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()
        
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
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        
        # 数据准备：创建dataloader
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        
        # 模型准备
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()
        
    def gp(self):
        """
        实现梯度惩罚功能
        """
        pass
        

```
#### 训练
##### 原理
- 训练判别器
##### 流程
- 
##### 实现
```python
    def train(self):
        """
        训练 generator 和 discriminator
        """
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.cuda()
                bs = imgs.size(0)

                # *********************
                # *    Train D-判别器  *
                # *********************
                z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()# 随机张量
                r_imgs = Variable(imgs).cuda()# 真是图片包装成Variable对象方便求梯度
                # 生成器生成假照片
                f_imgs = self.G(z)# 根据随机张量生成假图片
                r_label = torch.ones((bs)).cuda()# 真实图像标签置为1
                f_label = torch.zeros((bs)).cuda()# 假图像便签置为0


                # Discriminator前向传播
                r_logit = self.D(r_imgs)# 判别器计算真实图像的标签
                f_logit = self.D(f_imgs)

                """
                DISCRIMINATOR损失设置注意:
                    GAN:  loss_D = (r_loss + f_loss)/2
                    WGAN: loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                    WGAN-GP:
                        gradient_penalty = self.gp(r_imgs, f_imgs)
                        loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """
                # discriminator的损失: (评估 是否能区分真实图片和生成图片)
                #    生成fake->判别logit VS 0    &  real->判别logit VS 1 
                r_loss = self.loss(r_logit, r_label)# 计算真实损失
                f_loss = self.loss(f_logit, f_label)# 虚假损失
                loss_D = (r_loss + f_loss) / 2# 取损失的均值

                # Discriminator 反向传播
                self.D.zero_grad()
                loss_D.backward()# 反向传播
                self.opt_D.step()

                """
                设置 WEIGHT CLIP 注意:
                    WGAN: 使用以下code
                """
                # for p in self.D.parameters():
                #     p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])

                # *********************
                # *    Train G-生成器  *
                # *********************
                if self.steps % self.config["n_critic"] == 0:# 奇数训练生成器
                    # 生成一些假照片
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    f_imgs = self.G(z)

                    # Generator前向传播
                    f_logit = self.D(f_imgs)

                    """
                    生成器损失函数设置注意：
                        GAN: loss_G = self.loss(f_logit, r_label)
                        WGAN: loss_G = -torch.mean(self.D(f_imgs))
                        WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # 生成器损失(评估 生成图片和真实 是否很接近): 生成->判别logit VS 1
                    loss_G = self.loss(f_logit, r_label)

                    # Generator反向传播
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                    
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # 在训练过程中显示图片
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.G.train()

            if (e+1) % 5 == 0 or e == 0:
                # 保存checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))

        logging.info('Finish training')

```