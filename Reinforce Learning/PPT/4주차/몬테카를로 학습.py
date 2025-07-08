 import numpyas np
 import random
 from tqdmimport *
 from itertoolsimport cycle, count
 import gym
 from RLHelperimport *  # 헬퍼모듈
## 몬테카를로예측알고리즘
...............
 ................
 ## 프로즌레이크환경생성
env = gym.make('FrozenLake-v1')
 # 파라미터초기화
init_state= env.reset()
 goal_state= 15
 gamma = 0.99
 n_episodes= 2500
 P = env.env.P # 상태