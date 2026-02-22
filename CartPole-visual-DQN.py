import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]  # 4 عدد
action_size = env.action_space.n              # 2 اکشن

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

memory = deque(maxlen=2000)

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
episodes = 200

def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        return torch.argmax(model(state)).item()

def train():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q = model(next_states).max(1)[0]
    target = rewards + gamma * next_q * (1 - dones)

    loss = loss_fn(q_vals, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep:03d} | Reward: {total_reward}")

env.close()