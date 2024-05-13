import torch
from torch import nn
import torch.nn.functional as F
from collections import deque

#define the model
class DQN(nn.Module):

    def __init__(self, state_size, action_size) -> None:
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,action_size)


    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#define the agent

class DQNagent:

    def __init__(self,state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.batch_size = 64
        self.memory = deque(maxlen=100000)
        self.model = DQN(state_size,action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.001)
        self.loss_fn = torch.nn.MSELoss()
        self.rewards = []

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)