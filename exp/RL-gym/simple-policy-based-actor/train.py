import os
from typing import List

import cv2
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt

action_dim = 2
input_dim = 4
num_episodes = 2000
lr = 0.01
gamma = 0.95


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.act = nn.ReLU()
        self.out = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


class ReinforcementAgent:
    def __init__(self, policy_net: nn.Module, env: gym.Env, verbose=True):
        self.env = env
        self.policy_net = policy_net
        self.verbose = verbose
        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=lr, )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=500)
        self.total_rewards_record = []

        if os.path.exists('model.pth'):
            print('Detected pre-trained model')
            self.policy_net.load_state_dict(torch.load('model.pth'))

    def plot_total_reward(self, num_episodes: int):
        x = range(num_episodes)
        y = self.total_rewards_record
        plt.plot(x, y, color='red')
        plt.xlabel('episode')
        plt.ylabel('total rewards')
        plt.title(f'Learning curve(epoch={num_episodes})')
        plt.savefig('learning curve.png')

    def play_many_episodes(self, num_episodes: int):
        max_reward = 0
        for episode in range(num_episodes):
            log_probs, rewards = agent.play_one_episode()
            total_r = sum(rewards)
            if total_r > max_reward:
                torch.save(self.policy_net.state_dict(), 'model.pth')
            self.total_rewards_record.append(total_r)
            self.finish_episode(log_probs, rewards)
            print(f"Episode {episode} Total rewards = {total_r}")

        self.plot_total_reward(num_episodes)

    def finish_episode(self, log_props: List[float], rewards: List[float]):
        returns = self.calc_returns(rewards)
        policy_loss = []
        self.opt.zero_grad()
        for G, log_prob in zip(returns, log_props):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.opt.step()
        # self.scheduler.step()

    def calc_returns(self, rewards: List[float]):
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-20)
        return returns

    def play_one_episode(self) -> (List[float], List[float]):
        state = self.env.reset()[0]
        rewards = []
        log_props = []
        tot_steps = 0
        while True:
            # 采取一个行动
            state = torch.tensor(state, dtype=torch.float)
            probs = self.policy_net(state)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_props.append(log_prob)

            if self.verbose:
                img = env.render()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Environment", img)
                cv2.waitKey(1)

            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

            tot_steps += 1
            if done or tot_steps > 10000:
                return log_props, rewards


if __name__ == '__main__':
    # 创建一个环境
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # 重置环境
    state = env.reset()

    policy_net = PolicyNet(input_dim, action_dim)

    # 训练智能体
    agent = ReinforcementAgent(policy_net, env)

    agent.play_many_episodes(num_episodes)

    # 关闭环境
    cv2.destroyAllWindows()
    env.close()
