import numpy as np
from tqdm import tqdm
import time
import os

from snake_env import SnakeEnv
from DQN_agent import DQNAgent

env = SnakeEnv()
agent = DQNAgent()

np.random.seed(1)

if not os.path.isdir("models"):
    os.makedirs("models")

# Training settings
EPISODES = 20_000
# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 1e-3
# More Settings
SHOW_PREVIEW = True
SHOW_EVERY = 1
STATS_EVERY = 50
MIN_AVG_REWARD_FOR_SAVE = 200

ep_rewards = []
step = 0

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    # Update tensorboard step
    agent.callback.step = episode

    # Restarting episode
    episode_reward = 0
    done = False
    obs = env.reset()

    while not done:
        # Choose action
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(obs))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        # Take a step
        newObs, reward, done = env.step(action)
        step += 1

        # Update total episode reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % SHOW_EVERY:
            env.render()

        # Add to memory buffer
        agent.update_replay_memory((obs, action, reward, newObs, done))

        # Train agent
        agent.train(done)

        # Update state
        obs = newObs

    # Keep track of episode rewards
    ep_rewards.append(episode_reward)

    # Log stats
    if not episode % STATS_EVERY:
        average_reward = np.mean(ep_rewards[-STATS_EVERY:])
        min_reward = min(ep_rewards[-STATS_EVERY:])
        max_reward = max(ep_rewards[-STATS_EVERY:])
        agent.callback.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model
        if average_reward >= MIN_AVG_REWARD_FOR_SAVE:
            agent.model.save(f"models/{agent.MODEL_NAME}{max_reward:_>6.2f}max{average_reward:_>6.2f}avg{min_reward:_>6.2f}min{int(time.time())}.model")
            
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, MIN_EPSILON)

