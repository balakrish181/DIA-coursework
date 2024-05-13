import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from game import Battle
from constants import *
import torch.optim.lr_scheduler as lr_scheduler

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
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
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.1)

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
        self.scheduler.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define training parameters
state_size = 9  # Adjust based on the state representation in the environment
action_size = 5  # Adjust based on the number of actions in the environment
num_episodes = 400
batch_size = 256

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

env = Battle()
# Training loop
metrics = []
scores=[]
bullets = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    score = 0
    n_bullet = 0
    while step < MAX_STEPS_PER_EPISODE:
        action = agent.act(state)
        next_state, reward, done, score,n_bullet = env.rl_space(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay(batch_size)
        step +=1
    print("Episode:", episode + 1, "Total Reward:", total_reward,'Bullets fired: ',n_bullet, 'Score: ', score)
    metrics.append(total_reward)
    scores.append(score)
    bullets.append(n_bullet)

    if episode%100 == 0:
        torch.save(agent.model.state_dict(), "dqn_model_reward2.pth")



# Save the trained model
torch.save(agent.model.state_dict(), "dqn_model_reward2.pth")

import pandas as pd
# Assuming rewards and scores are lists or arrays of equal length
data = {'rewards': metrics,'bullets_fired': bullets, 'scores': scores}
df = pd.DataFrame(data)
df.to_csv('train_metrics_reward2.csv', index=False)

# Define testing parameters

print('-------------------------------------------------------')

print('Testing phase')

num_test_episodes = 100

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

env.close_turtle()

import pandas as pd
# Assuming test_rewards and test_scores are lists or arrays of equal length
data = {'test_rewards': test_rewards,'bullets_fired': test_bullets, 'test_scores': test_scores}
df = pd.DataFrame(data)
df.to_csv('test_metrics_reward2.csv', index=False)


print('----------------------------------------------------------')
