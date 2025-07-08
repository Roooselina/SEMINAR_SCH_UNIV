from gym import spaces
import numpy as np
import gym

# {상태 : {행동: [(전이확률, 다음상태, 보상, 완료 여부),
#                (전이확률, 다음상태, 보상, 완료 여부]}}
'''
 ㅁ      ㅁ - sleep
 facebook|           \
 |
 ㅁ -    ㅁ -   ㅁ -    ㅁ(pass)
class1 class2 class3
         |
       \ ㅁ-pup /
'''
#class 1을 0으로 한다.

''' 
1 5
0-2-3-4
  6
'''

MAP = {
    0: {
        0: [(0.5, 1, -1, False), (0.5, 2, -2, False)],
        1: [(0.5, 1, -1, False), (0.5, 2, -2, False)],
        2: [(0.5, 1, -1, False), (0.5, 2, -2, False)]
    },
    1: {
        0: [(0.9, 1, -1, False), (0.1, 0, -2, False)],
        1: [(0.9, 1, -1, False), (0.1, 0, -2, False)],
        2: [(0.9, 1, -1, False), (0.1, 0, -2, False)]
    },
    2: {
        0: [(0.8, 3, -2, False), (0.2, 5, 0, True)],
        1: [(0.8, 3, -2, False), (0.2, 5, 0, True)],
        2: [(0.8, 3, -2, False), (0.2, 5, 0, True)]
    },
    3: {
        0: [(0.6, 3, -2, False), (0.4, 6, 1, False)],
        1: [(0.6, 3, -2, False), (0.4, 6, 1, False)],
        2: [(0.6, 3, -2, False), (0.4, 6, 1, False)]
    },
    4: {
        0: [(1, 5, 10, True)],
        1: [(1, 5, 10, True)],
        2: [(1, 5, 10, True)],
    },
    6: {
        0: [(0.4, 2, -2, False), (0.2, 0, -2, False), (0.4, 3, -2, False)],
        1: [(0.4, 2, -2, False), (0.2, 0, -2, False), (0.4, 3, -2, False)],
        2: [(0.4, 2, -2, False), (0.2, 0, -2, False), (0.4, 3, -2, False)]
    }
}



class CustomEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3) # 0,1,2 랜덤 선택
        self.observation_space = spaces.Discrete(7)
        self.transition_probabilities = MAP
        self.current_state = None
        self.step_count = 0

    def reset(self):
        self.current_state = 0
        self.step_count = 0
        return self.current_state

    def step(self, action):
        # if self.current_state not in self.transition_probabilities or action not in self.transition_probabilities[self.current_state]:
        #     return self.current_state, 0, True, {}
        self.current_state not in self.transition_probabilities or action not in self.transition_probabilities[self.current_state]
        transition_info = self.transition_probabilities[self.current_state][action]
        choice_probabilities = [info[0] for info in transition_info]
        idx = np.random.choice(len(transition_info), p=choice_probabilities)
        prob, next_state, reward, done = transition_info[idx]
        self.current_state = next_state
        self.step_count += 1

        return next_state, reward, done, {"prob": prob}


env_name = 'CustomEnv-v0'
gym.register(env_name, entry_point=lambda: CustomEnv())
env = gym.make(env_name)

for e in range(10):
    state = env.reset()
    reward_sum = 0

    for t in range(500):
        action = env.action_space.sample()
        next_state, reward, done, prob = env.step(action)
        reward_sum += reward
        print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}, Probability: {prob}")

        if done:
            print(f'Episode finished after {t + 1} timesteps, total reward: {reward_sum}')
            break
        state = next_state
