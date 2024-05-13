import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from game import Battle
from constants import *


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for state, action, reward, next_state, done in minibatch:
            states_batch.append(state)
            actions_batch.append(action)
            rewards_batch.append(reward)
            next_states_batch.append(next_state)
            dones_batch.append(done)

        states_batch = torch.tensor(states_batch, dtype=torch.float32).to(self.device)
        actions_batch = torch.tensor(actions_batch, dtype=torch.int64).to(self.device)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(self.device)
        next_states_batch = torch.tensor(next_states_batch, dtype=torch.float32).to(self.device)
        dones_batch = torch.tensor(dones_batch, dtype=torch.bool).to(self.device)

        future_q_values = self.model(next_states_batch)
        target = rewards_batch + (1.0 - dones_batch.to(torch.float32)) * self.gamma * torch.max(future_q_values, dim=1)[0]
        target_f = self.model(states_batch).clone().detach()
        target_f[range(batch_size), actions_batch.squeeze()] = target
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(states_batch), target_f)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Define training parameters
state_size = 9  # Adjust based on the state representation in the environment
action_size = 5  # Adjust based on the number of actions in the environment
num_episodes = 500
batch_size = 64

agent = DQNAgent(state_size, action_size)

agent.model.load_state_dict(torch.load('D:\dia_experiment_dqn\sensitivity_exp\dqn_model_reward3.pth'))


num_test_episodes = 1








diff_init = 30




import numpy as np



mean_rewards = []
mean_scores = []
mean_bullets = []


import numpy as np

for each in range(diff_init):

    tar_pos = (120,120)
    agent_pos = (0,0)

    env = Battle(random_init = False,target_pos=tar_pos,agent_pos=agent_pos)
    test_rewards = []
    test_scores = []
    test_bullets = []
    # Testing loop
    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        score=0
        test_bullet =0 
        

        while step < MAX_STEPS_PER_EPISODE:
            action = agent.act(state)
            next_state, reward, done, score,test_bullet = env.rl_space(action)
            state = next_state
            total_reward += reward
            step += 1
            #if done:
                #break
        print("Test Episode:", episode + 1, "Total Reward:", total_reward,'Bullets fired: ',test_bullet, 'Score: ', score)
        test_rewards.append(total_reward)
        test_scores.append(score)
        test_bullets.append(test_bullet)
    
    mean_reward = np.mean(test_rewards)
    mean_score = np.mean(test_scores)
    mean_bullet = np.mean(test_bullets)
    mean_rewards.append(mean_reward)
    mean_scores.append(mean_score)
    mean_bullets.append(mean_bullet)

    #env.close_turtle()

import pandas as pd
# Assuming test_rewards and test_scores are lists or arrays of equal length
data = {'test_rewards': mean_rewards,'bullets_fired': mean_bullets, 'test_scores': mean_scores}
df = pd.DataFrame(data)
df.to_csv('test_metrics_same_init.csv', index=False)


