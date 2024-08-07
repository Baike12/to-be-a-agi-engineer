from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import typing as typ
from torch.distributions import Categorical
# from tqdm.auto import tqdm
from tqdm import tqdm # 下载notebook 仍能显示
import gym
import random
import os
from rich.console import Console
import warnings
print("gym.__version__=", gym.__version__)
NOTEBOOK_SEED = 543
def all_seed(seed=6666, env=None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
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


cs = Console()
def gym_env_desc(env_name):
    """
    对环境的简单描述
    """
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    cs.print("observation_space:\n\t", env.observation_space)
    cs.print("action_space:\n\t", env.action_space)
    try:
        action_shape = env.action_space.n
        action_type = '离散'
        extra=''
    except Exception as e:
        action_shape = env.action_space.shape
        low_ = env.action_space.low[0]  # 连续动作的最小值
        up_ = env.action_space.high[0]  # 连续动作的最大值
        extra=f'<{low_} -> {up_}>'
        action_type = '连续'
    print(f'[ {env_name} ](state: {state_shape},action: {action_shape}({action_type} {extra}))')
    return

# env_name= 'LunarLander-v2'
# gym_env_desc(env_name)
# env = gym.make(env_name)
# all_seed(seed=NOTEBOOK_SEED, env=env)
# env.reset()
# #
# img = plt.imshow(env.render(mode='rgb_array')) # 如果gym.__version__==0.26.2: 只需要 plt.imshow(env.render())
# done = False
# while not done:
#     a = env.action_space.sample()
#     n_state, reward, done, _ = env.step(a)
#     img.set_data(env.render(mode='rgb_array')) # 如果gym.__version__==0.26.2: 只需要 img.set_data(env.render())
#     display.display(plt.gcf())
#     display.clear_output(wait=True)


# 假设 env 已经被初始化
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

# random_show()
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

from torch.optim.lr_scheduler import StepLR
from typing import List, Dict, AnyStr
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
        action_prob = self.policy_net(torch.FloatTensor(state).to(self.device))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        return action.detach().cpu().item()

    def batch_update(self, batch_episode: List[Dict[AnyStr, List]]):
        for transition_dict in batch_episode:
            self.update(transition_dict)
        self.sche.step()

    def update(self, transition_dict):
        reward_list = transition_dict["rewards"]
        state_list = transition_dict["states"]
        action_list = transition_dict["actions"]

        if self.normalize_reward:
            reward_list = (np.array(reward_list)-np.mean(reward_list))/(np.std(reward_list)+1e-9)
        Rt = 0
        self.opt.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).unsqueeze(0).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action.long()))
            Rt = self.gamma * Rt + reward
            loss = -log_prob * Rt
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

def train_on_policy(
    agent,
    env,
    num_batch=450,
    random_batch=2,
    episode_per_batch=3,
    episode_max_step=300,
    save_mdoel_dir= r'D:\ML\models'
):
    """
    on policy 强化学习算法学习简单函数
    params:
        agent: 智能体
        env: 环境
        random_batch: 前N个batch用random Agent收集数据
        num_batch: 训练多少个batch
        episode_per_batch： 一个batch下多少个episode
        episode_max_step: 每个episode最大步数
        save_mdoel_dir: 模型保存的文件夹
    """
    EPISODE_PER_BATCH = episode_per_batch
    NUM_BATCH = num_batch
    RANDOM_BATCH = random_batch
    MAX_STEP = episode_max_step
    avg_total_rewards, avg_final_rewards, avg_total_steps = [], [], []
    agent.train()
    tq_bar = tqdm(range(RANDOM_BATCH + NUM_BATCH))
    recent_best = -np.inf
    batch_best = -np.inf
    all_rewards = []
    for batch in tq_bar:
        tq_bar.set_description(f"[ {batch+1}/{NUM_BATCH} ]")
        batch_recordes = []
        total_rewards = []
        total_steps = []
        final_rewards = []
        for ep in range(EPISODE_PER_BATCH):
            rec_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            total_reward, total_step = 0, 0
            while True:
                a = agent.policy(state)
                if batch < RANDOM_BATCH:
                    a = env.action_space.sample()

                n_state, reward, done, _ = env.step(a)
                # 收集每一步的信息
                rec_dict['states'].append(state)
                rec_dict['actions'].append(a)
                rec_dict['next_states'].append(n_state)
                rec_dict['rewards'].append(reward)
                rec_dict['dones'].append(done)
                state = n_state
                total_reward += reward
                total_step += 1
                if done or total_step > MAX_STEP:
                    # 一个episode结束后 收集相关信息
                    final_rewards.append(reward)
                    total_steps.append(total_step)
                    total_rewards.append(total_reward)
                    all_rewards.append(total_reward)
                    batch_recordes.append(rec_dict)
                    break

        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_step = sum(total_steps) / len(total_steps)
        recent_batch_best = np.mean(all_rewards[-10:])
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        avg_total_steps.append(avg_total_step)
        # 在进度条后面显示关注的信息
        tq_bar.set_postfix({
            "Total": f"{avg_total_reward: 4.1f}",
            "Recent": f"{recent_batch_best: 4.1f}",
            "RecentBest": f"{recent_best: 4.1f}",
            "Final": f"{avg_final_reward: 4.1f}",
            "Steps": f"{avg_total_step: 4.1f}"})
        agent.batch_update(batch_recordes)
        if avg_total_reward > batch_best and (batch > 4 + RANDOM_BATCH):
            batch_best = avg_total_reward
            agent.save_model(save_mdoel_dir + "_batchBest")
        if recent_batch_best > recent_best and (batch > 4 + RANDOM_BATCH):
            recent_best = recent_batch_best
            agent.save_model(save_mdoel_dir)

    return avg_total_rewards, avg_final_rewards, avg_total_steps

def trainCar():
    env_name= 'CartPole-v1'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    all_seed(seed=NOTEBOOK_SEED, env=env)
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=0.01,
        gamma=0.8,
        stepLR_step_size=80
    )
    avg_total_rewards, avg_final_rewards, avg_total_steps = train_on_policy(
        agent, env,
        num_batch=160,
        random_batch=2,
        episode_per_batch=5,
        episode_max_step=300,
        save_mdoel_dir=r'D:\ML\models'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].plot(avg_total_rewards, label='Total Rewards')
    axes[0].plot(avg_final_rewards, label='Final Rewards')
    axes[0].set_title(f'{env_name} - Rewards Curve')
    axes[0].legend()
    axes[1].plot(avg_total_steps, label='steps')
    axes[1].set_title(f'{env_name} - Steps Curve')
    axes[1].legend()
    plt.show()


def trainLL():
    env_name = 'LunarLander-v2'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    all_seed(seed=NOTEBOOK_SEED, env=env)
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=0.001,
        gamma=0.99,
        normalize_reward=True,
        stepLR_step_size = 150,
        stepLR_gamma = 0.75
    )

    avg_total_rewards, avg_final_rewards, avg_total_steps = train_on_policy(
        agent, env,
        num_batch=800,
        random_batch=3,
        episode_per_batch=3,
        episode_max_step=300,
        save_mdoel_dir='./check_point_LunarLander_REINFORCE'
    )


    # ### 训练结果查看
    # 在训练过程中，我们记录了`avg_total_reward`，它表示更新策略网络训练过程中的平均总奖励。
    # 从理论上讲，如果`Agent`变得更好，则`avg_tal_reward`将增加。

    # In[15]:


    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].plot(avg_total_rewards, label='Total Rewards')
    axes[0].plot(avg_final_rewards, label='Final Rewards')
    axes[0].set_title(f'{env_name} - Rewards Curve')
    axes[0].legend()
    axes[1].plot(avg_total_steps, label='steps')
    axes[1].set_title(f'{env_name} - Steps Curve')
    axes[1].legend()
    plt.show()
torch.cuda.is_available()
trainLL()