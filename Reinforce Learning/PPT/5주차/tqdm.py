import numpy as np
from tqdm import tqdm

def decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes):
    alphas = np.linspace(init_alpha, min_alpha, int(n_episodes * alpha_decay_ratio))
    return np.concatenate((alphas, np.full(n_episodes - len(alphas), min_alpha)))

def td(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, n_episodes=500):
    nS = env.observation_space.n
    V = np.zeros(nS, dtype=np.float64)
    
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {state: [] for state in range(nS)}
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    
    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        state = state[0]
        
        while not done:
            action = pi(state)
            next_state, reward, done, _, _ = env.step(action)
            
            td_target = reward + gamma * V[next_state] * (not done)
            targets[state].append(td_target)
            
            td_error = td_target - V[state]
            V[state] += alphas[e] * td_error
            
            state = next_state
        
        V_track[e] = V
    
    return V, V_track, targets


