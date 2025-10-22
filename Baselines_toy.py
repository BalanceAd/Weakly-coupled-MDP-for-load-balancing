import numpy as np
import pandas as pd
import random
from itertools import product
import matplotlib.pyplot as plt

# Simulation function for two policies: shortest-queue and round-robin
def simulate_performance(lambda_rate, policy_type, mu_vm=1.0, mu_cont=1.0,
                         C_vm=1, C_cont=1, C_block=10.0,
                         T_end=2000.0, T_warm=200.0):
    # Initialize state and simulation variables
    state = [0,0,0,0,0,0]  # (q_vm1, q_c1_1, q_c1_2, q_vm2, q_c2_1, q_c2_2)
    time = 0.0
    sum_queue_lengths_time = 0.0
    num_arrivals = 0
    num_blocked = 0
    rr_pointer = 0  # for round-robin policy

    # Helper to choose action
    def get_action(state):
        nonlocal rr_pointer
        if policy_type == 'shortest':
            total1 = state[0] + state[1] + state[2]
            total2 = state[3] + state[4] + state[5]
            return 1 if total1 < total2 else 2
        elif policy_type == 'round-robin':
            action = (rr_pointer % 2) + 1
            rr_pointer += 1
            return action
        else:
            raise ValueError("Unknown policy type")

    while time < T_end:
        # Calculate event rates
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

        # Time to next event
        dt = random.expovariate(total_rate)
        time += dt

        # Accumulate queue lengths after warm-up
        if time >= T_warm:
            sum_queue_lengths_time += sum(state) * dt

        # Determine event
        r = random.uniform(0, total_rate)
        cumulative = 0
        for event, rate in rates.items():
            cumulative += rate
            if r <= cumulative:
                chosen_event = event
                break

        # Process event
        if chosen_event == 'arrival':
            if time >= T_warm:
                num_arrivals += 1
            a = get_action(state)
            idx_vm = 2*(a-1)
            if state[idx_vm] < C_vm:
                state[idx_vm] += 1
            else:
                # VM queue full
                if state[idx_vm+1] < C_cont:
                    state[idx_vm+1] += 1
                elif state[idx_vm+2] < C_cont:
                    state[idx_vm+2] += 1
                else:
                    if time >= T_warm:
                        num_blocked += 1

        elif chosen_event in ('vm1','vm2'):
            idx_vm = 0 if chosen_event == 'vm1' else 3
            if state[idx_vm] > 0:
                conts = state[idx_vm+1:idx_vm+3]
                if sum(conts) < 2:
                    state[idx_vm] -= 1
                    for k in range(2):
                        if state[idx_vm+1+k] < C_cont:
                            state[idx_vm+1+k] += 1
                            break

        else:  # container service
            cont_map = {'cont1':1,'cont2':2,'cont3':4,'cont4':5}
            idx_cont = cont_map[chosen_event]
            if state[idx_cont] > 0:
                state[idx_cont] -= 1

    # Compute performance metrics
    duration = T_end - T_warm
    L = sum_queue_lengths_time / duration
    blocking_prob = num_blocked / num_arrivals if num_arrivals > 0 else np.nan
    lambda_eff = lambda_rate * (1 - blocking_prob)
    W = L / lambda_eff if lambda_eff > 0 else np.nan
    holding_cost_rate = L
    blocking_cost_rate = (num_blocked * C_block) / duration
    average_cost_rate = holding_cost_rate + blocking_cost_rate

    return L, blocking_prob, lambda_eff, W, holding_cost_rate, blocking_cost_rate, average_cost_rate
