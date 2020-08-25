import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import gym
import random


# critic network for evaluating actions
class Critic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# actor network for predicting actions
class Actor(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# agent class td3 algorithm
class Agent:
    def __init__(self, env, max_mem=100000):
        self.gamma = 0.99
        self.tau = 0.005
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        input_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        # πφ
        self.actor = Actor(input_dim, self.n_actions)
        # πφ'
        self.target_actor = Actor(input_dim, self.n_actions)
        # Qθ1, Qθ2
        self.critic1 = Critic(input_dim, self.n_actions)
        self.critic2 = Critic(input_dim, self.n_actions)
        # Qθ'1, Qθ'2
        self.target_critic1 = Critic(input_dim, self.n_actions)
        self.target_critic2 = Critic(input_dim, self.n_actions)
        self.memory = deque(maxlen=max_mem)
        self.batch_size = 64
        self.replace = 0
        self.train_step = 0
        self.update_actor_int = 2
        self.noise = 0.1

    # selecting action
    def select_action(self, obs):
        # action predicted by actor network (policy)
        obs = torch.tensor(obs, dtype=torch.float32).cuda()
        action = self.actor(obs).cuda()
        # add random exploration noise to action
        # a ∼ πφ(s) + exp noise,
        # exp noise ∼ N (0, σ)
        action = action + torch.tensor(np.random.normal(scale=self.noise), dtype=torch.float32).cuda()
        # make sure the action is in range of max and min actions od our env
        action = torch.clamp(action, *self.min_action, *self.max_action)
        # make returned action suitable for gym
        return action.cpu().detach().numpy()

    def store_trajectory(self, state, action, reward, next_state, done):
        # update replay buffer by storing transitions made by agent
        # store(s, a, r, s0) in B
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        # make sure replay buffer has enough data to sample
        if len(self.memory) < self.batch_size:
            return
        # # sample a random minibatch of N transitions (si, ai, ri, si+1) from R
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], \
                                                      [i[3] for i in batch], [i[4] for i in batch]
        # convert to tensors
        states = torch.tensor(states, dtype=torch.float32).cuda()
        actions = torch.tensor(actions, dtype=torch.float32).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
        next_states = torch.tensor(next_states, dtype=torch.float32).cuda()
        done = torch.tensor(done, dtype=torch.long).cuda()
        # target action  with random exploration noise
        # a~ ← πφ0' (s') ∼ clip(N (0, σ˜), −c, c)
        target_action = self.target_actor(next_states)
        target_action = target_action + torch.clamp(torch.tensor(np.random.normal(scale=self.noise)), -0.5, 0.5)
        target_action = torch.clamp(target_action, *self.min_action, *self.max_action)
        next_q1 = self.target_critic1(next_states, target_action)
        next_q2 = self.target_critic2(next_states, target_action)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        next_q1[done] = 0.0
        next_q2[done] = 0.0
        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)
        min_q = torch.min(next_q1, next_q2)
        # target
        # y ← r + γ min i=1,2 Qθ'i(s', a~)
        y = rewards + self.gamma * min_q
        y = y.view(self.batch_size, 1)
        # Update critics θi ← argminθi N−1 Σ(y−Qθi(s, a))2
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        loss1 = nn.MSELoss()(q1, y)
        loss2 = nn.MSELoss()(q2, y)
        loss = loss1 + loss2
        loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        self.train_step += 1
        if not self.train_step % self.update_actor_int:
            # Update actor φ by the deterministic policy gradient
            # ∇φJ(φ) = N −1 Σ∇aQθ1(s, a)|a=πφ(s)∇φπφ(s)
            self.actor.optimizer.zero_grad()
            actor_loss = self.critic1(states, self.actor(states))
            actor_loss = -torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()
            self.update_target_networks()
        else:
            return

    # function for updating target networks
    def update_target_networks(self):
        # get parameters of networks and make dictionaries from them
        actor = dict(self.actor.named_parameters())
        critic1 = dict(self.critic1.named_parameters())
        critic2 = dict(self.critic2.named_parameters())
        target_actor = dict(self.target_actor.named_parameters())
        target_critic1 = dict(self.target_critic1.named_parameters())
        target_critic2 = dict(self.target_critic2.named_parameters())
        # update target network parameters
        # θ'i ← τθi + (1 − τ)θ'i
        # φ'i ← τφ + (1 − τ)φ'
        for param in actor:
            actor[param] = self.tau * actor[param].clone() + (1 - self.tau) * target_actor[param].clone()
        for param in critic1:
            critic1[param] = self.tau * critic1[param].clone() + (1 - self.tau) * target_critic1[param].clone()
        for param in critic2:
            critic2[param] = self.tau * critic2[param].clone() + (1 - self.tau) * target_critic2[param].clone()

        self.target_actor.load_state_dict(actor)
        self.target_critic1.load_state_dict(target_critic1)
        self.target_critic2.load_state_dict(target_critic2)

    def save_models(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.target_actor.state_dict(), filename + "_target_actor")
        torch.save(self.target_critic1, filename + "_target_critic1")
        torch.save(self.target_critic2, filename + "_target_critic2")

    def load_models(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.target_actor.load_state_dict(torch.load(filename + "_target_actor"))
        self.target_critic1.load_state_dict(filename + "_target_critic1")
        self.target_critic2.load_state_dict(filename + "_target_critic2")


env = gym.make("MountainCarContinuous-v0")
agent = Agent(env)
num_episodes = 100
for episode in range(num_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        agent.store_trajectory(state, action, reward, next_state, done)
        agent.train()
    agent.save_models("model")
    print(f"Episode {episode}, Score {score}")

