import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import torch.optim.lr_scheduler as lr_scheduler
import random


#define the model
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
