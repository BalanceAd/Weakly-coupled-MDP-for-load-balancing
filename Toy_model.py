import numpy as np
import pandas as pd
from itertools import product
import random
import matplotlib.pyplot as plt


# Define parameters
lambda_rate = 1.0
mu_vm = 1.0
mu_cont = 1.0
C_vm = 1
C_cont = 1
C_block = 10.0
beta = 0.99  # discount factor

# Uniformization rate
Lambda_tot = lambda_rate + 2 * mu_vm + 4 * mu_cont

# Enumerate states
states = list(product([0,1], repeat=6))
state_to_idx = {s: i for i, s in enumerate(states)}
n_states = len(states)
actions = [1, 2]

# Cost function
def cost(state, a):
    hold = sum(state)
    q_vm = state[2*(a-1)]
    q_cont = state[2*(a-1)+1:2*(a-1)+3]
    block = 1 if (q_vm == C_vm and sum(q_cont) == 2) else 0
    return hold + C_block * block

# Next-state generator
def next_states_probs(state, a):
    probs = {}
    s = list(state)
    q_vm = s[2*(a-1)]
    q_cont = s[2*(a-1)+1:2*(a-1)+3]
    # Arrival
    if q_vm < C_vm:
        s_arr = s.copy()
        s_arr[2*(a-1)] += 1
    else:
        if sum(q_cont) < 2:
            s_arr = s.copy()
            for k in range(2):
                if s_arr[2*(a-1)+1+k] < C_cont:
                    s_arr[2*(a-1)+1+k] += 1
                    break
        else:
            s_arr = s.copy()
    probs[tuple(s_arr)] = lambda_rate / Lambda_tot

    # VM-level service
    for idx_vm in [0, 3]:
        s_serv = list(state)
        if state[idx_vm] > 0:
            conts = s_serv[idx_vm+1:idx_vm+3]
            if sum(conts) < 2:
                s_serv[idx_vm] -= 1
                for k in range(2):
                    if s_serv[idx_vm+1+k] < C_cont:
                        s_serv[idx_vm+1+k] += 1
                        break
        probs[tuple(s_serv)] = probs.get(tuple(s_serv), 0) + mu_vm / Lambda_tot

    # Container-level service
    for idx_cont in [1, 2, 4, 5]:
        s_serv = list(state)
        if state[idx_cont] > 0:
            s_serv[idx_cont] -= 1
        probs[tuple(s_serv)] = probs.get(tuple(s_serv), 0) + mu_cont / Lambda_tot

    return probs

# Build P and c
P = {a: np.zeros((n_states, n_states)) for a in actions}
c = {a: np.zeros(n_states) for a in actions}
for s in states:
    i = state_to_idx[s]
    for a in actions:
        c[a][i] = cost(s, a)
        for sp, p in next_states_probs(s, a).items():
            P[a][i, state_to_idx[sp]] = p

# Policy iteration with iterative evaluation
policy = np.ones(n_states, dtype=int)
policy_stable = False
tolerance = 1e-6
max_eval_iter = 1000

while not policy_stable:
    # Policy evaluation (iterative)
    V = np.zeros(n_states)
    for _ in range(max_eval_iter):
        V_next = np.zeros(n_states)
        for i in range(n_states):
            a = policy[i]
            V_next[i] = c[a][i] + beta * P[a][i].dot(V)
        if np.max(np.abs(V_next - V)) < tolerance:
            V = V_next
            break
        V = V_next

    # Policy improvement
    policy_stable = True
    for i in range(n_states):
        Q_vals = [c[a][i] + beta * P[a][i].dot(V) for a in actions]
        best_a = actions[int(np.argmin(Q_vals))]
        if best_a != policy[i]:
            policy[i] = best_a
            policy_stable = False

# Display results
df = pd.DataFrame({
    'state': states,
    'optimal_action': policy
})
import ace_tools as tools; tools.display_dataframe_to_user(name="Toy Model Optimal Policy", dataframe=df)


#The simulation for varying arrival rates from 1 to 10
# Simulation function (as before)
def simulate_performance(lambda_rate, mu_vm=1.0, mu_cont=1.0,
                         C_vm=1, C_cont=1, C_block=10.0,
                         T_end=2000.0, T_warm=200.0):
    def policy(state):
        total1 = state[0] + state[1] + state[2]
        total2 = state[3] + state[4] + state[5]
        return 1 if total1 < total2 else 2

    state = [0,0,0,0,0,0]
    time = 0.0
    sum_queue_lengths_time = 0.0
    num_arrivals = 0
    num_blocked = 0

    while time < T_end:
        rates = {
            'arrival': lambda_rate,
            'vm1': mu_vm if state[0] > 0 else 0.0,
            'vm2': mu_vm if state[3] > 0 else 0.0,
            'cont1': mu_cont if state[1] > 0 else 0.0,
            'cont2': mu_cont if state[2] > 0 else 0.0,
            'cont3': mu_cont if state[4] > 0 else 0.0,
            'cont4': mu_cont if state[5] > 0 else 0.0
        }
        total_rate = sum(rates.values())
        if total_rate == 0:
            break

        dt = random.expovariate(total_rate)
        time += dt

        if time >= T_warm:
            sum_queue_lengths_time += sum(state) * dt

        r = random.uniform(0, total_rate)
        cumulative = 0.0
        for event, rate in rates.items():
            cumulative += rate
            if r <= cumulative:
                chosen_event = event
                break

        if chosen_event == 'arrival':
            if time >= T_warm:
                num_arrivals += 1
            a = policy(tuple(state))
            idx_vm = 2*(a-1)
            if state[idx_vm] < C_vm:
                state[idx_vm] += 1
            else:
                if state[idx_vm+1] < C_cont:
                    state[idx_vm+1] += 1
                elif state[idx_vm+2] < C_cont:
                    state[idx_vm+2] += 1
                else:
                    if time >= T_warm:
                        num_blocked += 1

        elif chosen_event in ('vm1','vm2'):
            idx_vm = 0 if chosen_event=='vm1' else 3
            if state[idx_vm] > 0:
                conts = state[idx_vm+1:idx_vm+3]
                if sum(conts) < 2:
                    state[idx_vm] -= 1
                    for k in range(2):
                        if state[idx_vm+1+k] < C_cont:
                            state[idx_vm+1+k] += 1
                            break
        else:
            cont_map = {'cont1':1,'cont2':2,'cont3':4,'cont4':5}
            idx = cont_map[chosen_event]
            if state[idx] > 0:
                state[idx] -= 1

    duration = T_end - T_warm
    L = sum_queue_lengths_time / duration
    blocking_prob = num_blocked / num_arrivals if num_arrivals>0 else np.nan
    lambda_eff = lambda_rate * (1 - blocking_prob)
    W = L / lambda_eff if lambda_eff>0 else np.nan
    holding_cost_rate = L
    blocking_cost_rate = (num_blocked * C_block) / duration
    average_cost_rate = holding_cost_rate + blocking_cost_rate

    return L, blocking_prob, lambda_eff, W, holding_cost_rate, blocking_cost_rate, average_cost_rate
