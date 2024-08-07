## RL
### 回顾理论
- 一个智能体
#### 基本流程
##### 定义函数
##### 定义损失
- 奖励是立即的反馈，回报是整局的奖励总和
- 轨迹：状态和动作的组合序列$\tau={s_{1},a_{1},s_{2},a_{2}\dots}$ 
##### 优化
- 优化的目的是让回报最大
#### 和GAN的区别
- GAN的目的是调整生成器参数使得判别器输出更大，就是让判别器认为生成器的输出更像一个真实输出
	- GAN的判别器也是一个神经网络，可用梯度下降的反向传播
- RL的环境和奖励不是神经网络，无法用梯度下降
##### 既要倾向某个动作$a$ 此时损失为交叉熵 $e_{1}$  ，又要避免某个动作$a'$ 此时损失为$-e_{2}$ 
- 整体损失$\theta ^{*}=e_{1}-e_{2}$ 
#### 评价动作的标准
##### 即时间奖励
- 使用奖励而不是回报
- 会导致短视
##### 累积奖励
- $G_{t}=\sum_{i=t}^{N}r_{i}$ 
	-  t时刻的奖励是从t时刻到最后的奖励的总和
- 整个动作很长会导致复杂计算
##### 使用折扣因子
- $G_{t}'=\sum_{i=t}^{N}\gamma ^{i-t}r_{i}$ 
	- $\gamma<1$ 离t越远权重越小
- 适用于游戏这种近期才做比较影响奖励的情形
- 不好的动作奖励也会是正的
##### 减去基线
- $A_{i}=G_{i}'-b$ 
#### 整体流程
- 随机参数的智能体
- 和环境互动得到一系列状态-动作对s-a
- 计算这些s-a的奖励，评价s-a的好坏
- 根据好坏计算损失更新模型参数$\theta _{i-1}\to\theta _{i}$ 
##### RL的数据收集
- 在训练迭代中收集数据：因为智能体本身的参数会影响奖励
- 获取s-a，计算奖励A，更新参数$\theta$ ，使用新的参数获取s-a
##### 异策略学习
- 智能体agent1与环境互动的数据用来训练智能体agent2
##### 探索
- 加大智能体动作的随机性
#### Actor-Critic
##### Critic
- 价值函数$V_{\pi \theta}(s)$ ：输入状态，输出折扣累积奖励$V_{out}$ 
- 训练方法
	- 蒙特卡洛：玩完整局游戏
		- 看到sa，可以得到累积奖励$G_{a}'$ ，Critic输出应该接近累积奖励
	- 时序差分法
		- 看到${s _{t}, a_{t}, r_{t}, s _{t+1}}$ 就可以输出$V_{out}$ 
		- 因为$V_{\pi \theta}(s _{t})=\gamma V_{\pi \gamma}(s _{t+1})+r_{t}$ 
		- 流程
			- 把st和s（t+1）带到价值函数中得到$V_{\pi \theta}(s _{t})$ 和$V_{\pi \theta}(s _{t+1})$ 
			- 这两个价值函数之差应该接近$r_{t}$ 
##### 使用Critic训练Actor
- 此时Critic已经训练出来，给一个状态sa就能得到一个价值函数输出
- 把价值函数输出当做基线$A_{i}=G_{i}'-V_{\pi \theta}(s _{i})$ 
	- $G_{i}'$ 是当前a的累积奖励
	- 价值函数是状态为sa时不同a的情况下平均值，这些不同的动作a是随机采样的
	- 如果$A_{i}$  > 0，说明当前的a比随机采样的a要好
##### 优势Actor-Critic
- 执行$a_{t}$ 之后得到$r_{t}$ ，然后继续下一步，下一步有很多可能，这些可能的期望是$V_{\pi \theta}(s _{t+1})$ ，$r_{t}+V_{\pi \theta}(s _{t+1})$ 是在当前执行$a_{t}$ 得到的奖励，可以作为$G_{t}'$ 
- 这一步的奖励可以写成$V_{\pi \theta}(s _{t+1})+r_{t}-V_{\pi \theta}(s _{t})$ 
	- 前面部分是采取$a_{t}$ 之后的奖励，后一部分是随机采样奖励
##### 训练技巧
- Actor和Critic都是输入一个图像
	- Actor输出一个分数
	- Critic输出一个数值表示累积奖励
- 可以共用前几层的参数，因为前几层都在处理图像
### 环境描述
##### 原理
-  打印观察空间和动作空间
##### 实现
```python
def gym_env_desc(env_name):
    """
    对环境的简单描述
    """
    env = gym.make(env_name)# 创建环境
    state_shape = env.observation_space.shape# 获取状态空间
    cs.print("observation_space:\n\t", env.observation_space)# 打印状态空间信息
    cs.print("action_space:\n\t", env.action_space)# 打印动作空间信息
    try:
        action_shape = env.action_space.n# 获取离散空间大小
        action_type = '离散'
        extra=''
    except Exception as e:# 不是离散的
        action_shape = env.action_space.shape
        low_ = env.action_space.low[0]  # 连续动作的最小值
        up_ = env.action_space.high[0]  # 连续动作的最大值
        extra=f'<{low_} -> {up_}>'
        action_type = '连续'
    print(f'[ {env_name} ](state: {state_shape},action: {action_shape}({action_type} {extra}))')
    return 
```
### 随机agent测试
##### 原理
- 随机动作看看能不能精准降落 
##### 实现
```python
env = gym.make('LunarLander-v2')
gym_env_desc('LunarLander-v2')
observation = env.reset()

# 创建一个新的图形窗口
fig, ax = plt.subplots()
ax.axis('off')  # 关闭坐标轴

# 创建一个图像对象
img = ax.imshow(env.render(mode='rgb_array'))

def update_image():
    global observation
    a = env.action_space.sample()
    n_state, reward, done, _ = env.step(a)
    if not done:
        observation = n_state
    img.set_data(env.render(mode='rgb_array'))
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return done

def random_show():
# 开始动画循环
    try:
        while True:
            done = update_image()
            if done:
                break
            plt.pause(0.01)  # 暂停一小段时间以产生动画效果
    finally:
        env.close()

```
### 策略梯度
##### 功能与原理
- 策略是从状态到动作的概率分布$\pi(a|s)$ 
	- 状态s的情况下a的概率
- 状态价值函数$V(s)$ 是在状态s的情况下期望的回报
- 目标函数$J(\theta)=E_{\tau\sim \pi _{\theta}}[R(\tau)]$ 
	- 当参数是$\theta$  时选取的序列$\tau$ 这些序列的回报的期望
	- 期望越大越好
- 由于直接计算目标函数的梯度很难，一般使用样本估计梯度
	- $\nabla _{\theta}J(\theta)=\frac{1}{N}\sum_{i=1}^{N}R(\tau _{i})\nabla _{\theta}\log \pi _{\theta}(a_{i}|s _{i})$ 
		- 转换成求策略网络的对数的梯度，也就是策略梯度
- 网络返回动作空间概率，然后抽样一个动作 
- 输入状态张量，输出动作张量

##### 实现
```python
class PolicyGradientNet(nn.Module):# 策略网络
    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyGradientNet, self).__init__()
        self.q_net = nn.ModuleList([
            nn.ModuleDict({
                'linear': nn.Linear(state_dim, 32),
                'linear_activation': nn.Tanh()
            }),
            nn.ModuleDict({
                'linear': nn.Linear(32, 32),
                'linear_activation': nn.Tanh()
            }),
            nn.ModuleDict({
                'linear': nn.Linear(32, action_dim),
                'linear_activation': nn.Softmax(dim=-1)
            })
        ])
        
    def forward(self, x):
        for layer in self.q_net:
            x = layer['linear_activation'](layer['linear'](x))
        return x

class REINFORCE():
    def __init__(self, state_dim: int, action_dim: int, lr: float=0.001, gamma: float=0.9, 
                 stepLR_step_size:int = 200,
                 stepLR_gamma:float = 0.1,
                 normalize_reward:bool=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyGradientNet(state_dim, action_dim)
        self.policy_net.to(self.device)
        self.opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.stepLR_step_size = stepLR_step_size
        self.stepLR_gamma = stepLR_gamma
        self.sche = StepLR(self.opt, step_size=self.stepLR_step_size, gamma=self.stepLR_gamma)
        self.gamma = gamma
        self.normalize_reward = normalize_reward
        self.training = True
    
    def train(self):
        self.training = True
        self.policy_net.train()

    def eval(self):
        self.training = False
        self.policy_net.eval()
        
    @torch.no_grad()
    def policy(self, state):
        """
        sample action
        """
        action_prob = self.policy_net(torch.FloatTensor(state).to(self.device))# 计算动作概率
        action_dist = Categorical(action_prob)# 包装成Categorical对象，变成动作列表
        action = action_dist.sample()# 从动作列表中采样
        return action.detach().cpu().item()# 采样值转移到cpu后转换成numpy数组

    def batch_update(self, batch_episode: List[Dict[AnyStr, List]]):
        for transition_dict in batch_episode:#
            self.update(transition_dict)
        self.sche.step()# 调整学习率

    def update(self, transition_dict):# 更新策略网络参数
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        # 分数 normalize
        if self.normalize_reward:# 奖励归一化
            reward_list = (np.array(reward_list) - np.mean(reward_list)) / (np.std(reward_list) + 1e-9)
        Rt = 0
        self.opt.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).unsqueeze(0).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action.long()))
            # Rt = \sum_{i=t}^T \gamma ^ {i-t} r_i
            Rt = self.gamma * Rt + reward
            loss = -log_prob * Rt# 损失函数使得选择一个好动作（折扣回报高）时损失函数较小
            loss.backward()
        self.opt.step() 
        
    def save_model(self, model_dir):# 保存模型
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        file_path = os.path.join(model_dir, 'policy_net.ckpt')
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, model_dir):# 加载模型
        file_path = os.path.join(model_dir, 'policy_net.ckpt')
        self.policy_net.load_state_dict(torch.load(file_path))


```
### 训练agent
### 测试
### 训练lunarLander-v2
### 测试
### 服务器
### actor-critic
### 训练agent
### 最后测试
