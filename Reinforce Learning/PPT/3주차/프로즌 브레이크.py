import numpy as np
import random

import gym
from outputHelper import *

## 프로즌 레이크 환경 생성
env = gym.make('FrozenLake-v1')

 # 상태 저장
P = env.env.P
print(P)

init_state= env.reset(seed=123)     # 환경 초기화
goal_state= 15                      # 목표 상태

## 임의의 정책 생성
LEFT, DOWN, RIGHT, UP = range(4)
random_pi= lambda s: {
0:RIGHT, 1:LEFT, 2:DOWN, 3:UP,
4:LEFT, 5:LEFT, 6:RIGHT, 7:LEFT,
8:UP, 9:DOWN, 10:UP, 11:LEFT,
12:LEFT, 13:RIGHT, 14:DOWN, 15:LEFT
}[s]

improved_pi= policy_improvement(random_pi, P)

# 정책, 성공확률, 평균리턴값출력
print_policy(improved_pi, P)
print('성공확률{:.2f}%. 평균리턴값{:.4f}.'.format(
probability_success(env, random_pi, 
goal_state=goal_state)*100, mean_return(env, random_pi)))

 ## 정책 평가 수행
V = policy_improvement(random_pi, P, gamma=0.99)
 # 상태-가치함수출력
print_state_value_function(V, P, prec=4)

# 정책, 성공확률, 평균리턴값출력
print_policy(improved_pi, P)
print('성공확률{:.2f}%. 평균리턴값{:.4f}.'.format(probability_success(env, improved_pi, goal_state=goal_state)*100,    mean_return(env, improved_pi)))