from airsim_env import AirSimCarEnv
from dqn_agent import DQNAgent
import torch

env = AirSimCarEnv()
agent = DQNAgent(state_dim=4, action_dim=3)

num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
torch.save(agent.model.state_dict(), "dqn_model.pth")
