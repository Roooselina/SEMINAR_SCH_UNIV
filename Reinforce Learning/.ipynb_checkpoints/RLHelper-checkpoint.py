import numpy as np
import random
from itertools import cycle, count
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def mean_return(env, gamma, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(seed=123), False, 0
        state = state[0]               # s => (0, {'prob': 1})
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _, _ = env.step(pi(state))
            results[-1] += (gamma**steps * reward) # 할인율추가
            steps += 1
    return np.mean(results)


def rmse(x, y, dp=4):
    return np.round(np.sqrt(np.mean((x -y)**2)), dp)


plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
     'font.size': 24,
     'legend.fontsize': 20,
     'axes.titlesize': 28,
     'axes.labelsize': 24,
    'xtick.labelsize': 20,
     'ytick.labelsize': 20
 }
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)


def plot_value_function(title, V_track, V_true=None, log=False, limit_value=0.05, limit_items=5):
    np.random.seed(123)
    per_col = 25
    linecycler = cycle(["-", "--", ":", "-."])
    legends = []
    valid_values = np.argwhere(V_track[-1] > limit_value).squeeze()
    items_idxs = np.random.choice(valid_values, min(len(valid_values), limit_items), replace=False)

    if V_true is not None:
        for i, state in enumerate(V_track.T):
            if i not in items_idxs:
                continue
            if state[-1] < limit_value:
                continue
            label = 'v({})'.format(i)
            plt.axhline(y=V_true[i], color='k', linestyle='-', linewidth=1)
            plt.text(int(len(V_track) * 1.02), V_true[i] + .01, label)

    for i, state in enumerate(V_track.T):
        if i not in items_idxs:
            continue
        if state[-1] < limit_value:
            continue
        line_type = next(linecycler)
        label = 'V({})'.format(i)
        p, = plt.plot(state, line_type, label=label, linewidth=3)
        legends.append(p)

    legends.reverse()
    ls = []
    for loc, idx in enumerate(range(0, len(legends), per_col)):
        subset = legends[idx:idx + per_col]
        l = plt.legend(subset, [p.get_label() for p in subset], loc='center right', bbox_to_anchor=(1.25, 0.5))
        ls.append(l)

    [plt.gca().add_artist(l) for l in ls[:-1]]
    if log:
        plt.xscale('log')
    plt.title(title)
    plt.ylabel('State-value function')
    plt.xlabel('Episodes (log scale)' if log else 'Episodes')
    plt.show()


def plot_targets(targets, init_state, title):
    x = range(len(targets[init_state]))
    y = targets[init_state]
    label = 'v({})'.format(init_state)
    plt.axhline(y=V_true[init_state], color='k', linestyle='-', linewidth=1)
    plt.text(int(x[-1]*1.02), V_true[init_state]+.01, label)
    plt.scatter(x, y, c=np.array(targets[init_state]), cmap=plt.get_cmap('viridis'), alpha=0.4)
    plt.title(title)
    plt.ylabel('Target value')
    plt.xlabel('Estimate sequence number')
    plt.show()


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

def value_iteration(P, gamma=0.99, theta=1e-10):
    nS = len(P)
    nA = len(P[0])
    V = np.zeros(nS)
    Q = np.zeros((nS, nA))
    pi = np.zeros(nS, dtype=int)
    
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            Q[s] = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in P[s][a]]) for a in range(nA)]
            V[s] = max(Q[s])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    for s in range(nS):
        pi[s] = np.argmax(Q[s])
    
    return Q, V, pi

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