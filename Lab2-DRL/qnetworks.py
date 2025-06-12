# Implement Q-networks for the Deep Reinforcement Learning project
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
import wandb


class QNetwork(torch.nn.Module):
    def __init__(self, env, n_hidden=2, width=64):
        super(QNetwork, self).__init__()
        obs_dim = env.observation_space.shape[0] # Dimension of the observation space
        action_dim = env.action_space.n # Number of actions available in the environment

        # Create a sequence of hidden layers
        hidden_layers = [nn.Linear(obs_dim, width), nn.ReLU()]
        hidden_layers += [nn.Linear(width, width), nn.ReLU()] * (n_hidden - 1)
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width, action_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), actions, rewards, np.array(next_states), dones)

    def __len__(self):
        return len(self.buffer)




def train_dqn(env, run, episodes=1000, batch_size=64, gamma=0.99, lr=1e-3, tau=0.005, start_training=1000, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
    q_net = QNetwork(env) # Q-network for DQN
    target_net = QNetwork(env) # Target network for DQN
    target_net.load_state_dict(q_net.state_dict()) # Initialize target network with same weights as Q-network
    target_net.eval()  # Set target network to evaluation mode

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            total_steps += 1
            if random.random() < epsilon: # Epsilon-greedy action selection
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)#.unsqueeze(0)
                    action = q_net(state_tensor).argmax().item() # Select action with highest Q-value

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) > start_training:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states_tensor = torch.tensor(states, dtype=torch.float32)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                # Compute Q-values and target Q-values
                q_values = q_net(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    target_q_values = target_net(next_states_tensor).max(1, keepdim=True)[0]
                    target_q_values = rewards_tensor + (1 - dones_tensor) * gamma * target_q_values

                # Compute loss and optimize
                loss = nn.functional.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for param, target_param in zip(q_net.parameters(), target_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data) # Soft update of target network

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon

        # Wandb log
        log = {
            'total_reward': total_reward,
            'epsilon': epsilon,
        }

        run.log(log)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
