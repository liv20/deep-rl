import torch
from torch.distributions import Categorical
import numpy as np
import gym

from models.sequential import mlp
from algorithms.reward_calculation import compute_rtg

import yaml

with open('train_vpg.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)
# create environment
env = gym.make(config['env_name'])
observation_space = env.observation_space
action_space = env.action_space

# create policy
policy = mlp(observation_space.shape[0], action_space.n, 2, 32)
print(policy)

# create optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=config['lr'])

n_episode = 1
n_timestep = 1

running_rewards = []
while n_timestep < config['max_timesteps']:
    rewards, actions, states, dones = [], [], [], []

    state = env.reset()
    episode_timestep = 1
    while episode_timestep < config['max_episode_timesteps']:
        logits = policy(torch.tensor(state).unsqueeze(0).float())
        dist = Categorical(logits=logits)
        action = dist.sample()
    
        new_state, reward, done, info = env.step(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        state = new_state
        n_timestep += 1
        episode_timestep += 1

        if done:
            break

    rewards = np.array(rewards)
    returns = torch.tensor(compute_rtg(rewards, gamma=config['gamma']), dtype=torch.float32)

    states = torch.tensor(np.vstack(states)).float()
    actions = torch.tensor(actions)

    logits = policy(states)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)

    optimizer.zero_grad()
    loss = - torch.sum(log_probs * returns)
    loss.backward()
    optimizer.step()

    # calculate average return and print it out
    running_rewards.append(np.sum(rewards))
    if n_episode % 100 == 0:
        print("Episode: {:6d}\tTotal Return: {:6.2f}".format(n_episode, np.mean(np.array(running_rewards))))
        running_rewards = []
    n_episode += 1

env.close()


        