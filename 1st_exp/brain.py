import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from game import Battle  # Import the custom Battle environment
from constants import *  # Import constants like MAX_STEPS_PER_EPISODE
import torch.optim.lr_scheduler as lr_scheduler
from model import DQNAgent  # Import the DQNAgent class

# Define training parameters
state_size = 9  # Size of the state representation 
action_size = 5  # Number of possible actions 
num_episodes = 1000  # Number of training episodes
batch_size = 256  # Batch size for replay memory sampling

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Initialize the environment
env = Battle()

# Initialize metrics to track performance
metrics = []
scores = []
bullets = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()  # Reset environment and get initial state
    done = False
    total_reward = 0
    step = 0
    score = 0
    n_bullet = 0
    
    while step < MAX_STEPS_PER_EPISODE:
        action = agent.act(state)  # Select action based on current state
        next_state, reward, done, score, n_bullet = env.rl_space(action)  # Take action and observe results
        agent.remember(state, action, reward, next_state, done)  # Store experience in replay memory
        state = next_state
        total_reward += reward
        agent.replay(batch_size)  # Train the agent with a batch from replay memory
        step += 1
    
    # Print episode summary
    print("Episode:", episode + 1, "Total Reward:", total_reward, 'Bullets fired:', n_bullet, 'Score:', score)
    
    # Append metrics
    metrics.append(total_reward)
    scores.append(score)
    bullets.append(n_bullet)

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_model_reward1.pth")

# Save training metrics to CSV
import pandas as pd
data = {'rewards': metrics, 'bullets_fired': bullets, 'scores': scores}
df = pd.DataFrame(data)
df.to_csv('train_metrics_reward1.csv', index=False)

# Testing phase
print('-------------------------------------------------------')
print('Testing phase')

# Define testing parameters
num_test_episodes = 100

test_rewards = []
test_scores = []
test_bullets = []

# Testing loop
for episode in range(num_test_episodes):
    state = env.reset()  # Reset environment and get initial state
    done = False
    total_reward = 0
    step = 0
    score = 0
    test_bullet = 0

    while step < MAX_STEPS_PER_EPISODE:
        action = agent.act(state)  # Select action based on current state
        next_state, reward, done, score, test_bullet = env.rl_space(action)  # Take action and observe results
        state = next_state
        total_reward += reward
        step += 1
    
    # Print test episode summary
    print("Test Episode:", episode + 1, "Total Reward:", total_reward, 'Bullets fired:', test_bullet, 'Score:', score)
    
    # Append test metrics
    test_rewards.append(total_reward)
    test_scores.append(score)
    test_bullets.append(test_bullet)

# Close the environment
env.close_turtle()

# Save testing metrics to CSV
data = {'test_rewards': test_rewards, 'bullets_fired': test_bullets, 'test_scores': test_scores}
df = pd.DataFrame(data)
df.to_csv('test_metrics_reward1.csv', index=False)

print('----------------------------------------------------------')
