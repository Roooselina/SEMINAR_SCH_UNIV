import numpy as np
import random

## 정책출력

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='정책:'):
    print(title)
    arrs= {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols== 0: print("|")

## 정책 성공 확률 출력
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) 
    results = []
    for _ in range(n_episodes):
        s, done, steps = env.reset(seed=123), False, 0
        state = s[0]               # s => (0, {'prob': 1})
        while not done and steps < max_steps:
            state, _, done, _, _ = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)


## 평균 리턴값 출력
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123)
    results = []
    for _ in range(n_episodes):
        s, done, steps = env.reset(seed=123), False, 0
        state = s[0]               # s => (0, {'prob': 1})
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)


## 상태 가치 함수 출력
def print_state_value_function(V, P, n_cols=4, prec=3, title='상태-가치함수:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols== 0:
            print("|")

## 정책 평가 알고리즘
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10): # 정책, 환경의상태, 할인율, 세타(수렴값) 
    prev_V= np.zeros(len(P), dtype=np.float64)       # 이전 가치함수(k) 를0으로초기화
    # 수렴할 때까지 반복
    while True:
        V = np.zeros(len(P), dtype=np.float64)            # 현재 가치함수(k+1)를 0으로 초기화
        # 모든 상태들에 대해 반복
        for s in range(len(P)):
            # 현재 상태에서 가능한 전이에 대해 반복
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))      # 종료상태(done)인 경우엔 0
        if np.max(np.abs(prev_V-V)) < theta:    # 수렴한 경우 루프 빠져나옴
            break
        prev_V= V.copy()     # 현재상태를 이전 상태로 복사
    return V          # 계산된 가치함수 값을 리턴

## 정책 개선 알고라즘
def policy_improvement(V, P, gamma=1.0):            # 상태-가치함수, 환경의상태, 할인율인수
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)        # 상태-가치함수를 0으로 초기화
    # 모든상태들에반복
    for s in range(len(P)):
        # 현재 상태에서 모든 가능한 행동에 대해 반복
        for a in range(len(P[s])):
        # 현재상태, 행동에서 가능한 전이에 대해 반복
            for prob, next_state, reward, done in P[s][a]:
                print(V)
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))       # 종료상태(done)인경우엔0
            # 그리디 개선으로 정책을 개선
    new_pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi # 개선된 새로운 정책을 리턴
