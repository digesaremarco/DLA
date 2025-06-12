import numpy as np
import torch
from torch.distributions import Categorical

def select_action(env, obs, policy):

    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

def compute_returns(rewards, gamma):

    # Calculate discounted rewards in reverse order, then flip the array
    discounted_rewards = np.array([gamma**(i + 1) * r for i, r in enumerate(rewards)][::-1])
    total_returns = np.cumsum(discounted_rewards)

    return np.flip(total_returns, axis=0).copy()

# Given an environment and a policy, run it up to the maximum number of steps
def run_episode(env, policy, maxlen=500):

    # Collect just about everything
    observations = []
    actions = []
    log_probs = []
    rewards = []

    # Reset the environment and start the episode
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        # Advance the episode by executing the selected action
        (obs, reward, term, trunc, _) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break

    # Return just about everything
    return (observations, actions, torch.cat(log_probs), rewards)
