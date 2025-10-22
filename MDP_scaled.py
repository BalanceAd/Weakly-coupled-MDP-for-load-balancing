import numpy as np
import random
import pandas as pd
from math import inf
from collections import defaultdict
from dataclasses import dataclass



# -----------------------------
# Global model parameters
# -----------------------------
N        = 50           # number of VMs
K        = 2            # containers per VM (unit capacity each)
C_VM     = 3            # feeder queue capacity per VM
C_CT     = 1            # container capacity (0/1 busy per container)
MU_VM    = 2.0          # promotion rate
MU_CT    = 1.0          # per-container service rate
H_VM     = 1.0          # holding cost coefficient (VM queue)
H_CT     = 1.0          # holding cost coefficient (containers)
C_BLOCK  = 15.0         # blocking cost per rejected job
BETA     = 0.99         # discount factor for DP (uniformized chain)
OFFER_RATE = 15.0        # exogenous offer rate per VM (>= max λ/N in your sweeps)


# ------------------------------
# Offer rate auto tuner
# ------------------------------


def max_accept_rate_per_vm(offer_rate):
    # Solve at very negative theta (accept whenever feasible)
    theta_probe = -C_BLOCK
    V, policy, pi_ss, _ = policy_iteration_local(theta_probe, offer_rate=offer_rate)
    feas = np.array([accept_feasible(states[i]) for i in range(nS)], float)
    A = np.array([(policy[i]==1 and feas[i]==1.0) for i in range(nS)], float)
    return offer_rate * float(pi_ss @ A)

def alpha_at(theta, offer_rate):
    """
    Return the per-VM acceptance rate α(θ) [jobs / unit time] under the local offer-rate MDP
    at price θ and offer rate = offer_rate. Must satisfy 0 ≤ α(θ) ≤ offer_rate.
    """
    V, policy, pi_ss, _ = policy_iteration_local(theta, offer_rate=offer_rate)
    # Feasibility mask
    feas = np.array([accept_feasible(states[i]) for i in range(nS)], dtype=float)
    # Accept when (policy says ACCEPT) and (accept is feasible)
    A = np.array([(policy[i] == 1) and (feas[i] == 1.0) for i in range(nS)], dtype=float)
    # Per-VM accepted flow per unit time
    alpha_i = offer_rate * float(pi_ss @ A)
    return alpha_i

def autotune_offer_rate(lam, offer_rate_init=5.0, safety=1.15, max_tries=10):
    target = lam / N
    r = float(offer_rate_init)
    for _ in range(max_tries):
        amax = alpha_at(-C_BLOCK, r)  # accept whenever feasible
        if amax >= safety * target:
            return r
        r *= 2.0                     # geometric growth
    return r

# -----------------------------
# Utilities for verifying Schur
# convexity of value fction
# -----------------------------

def canonical_key(q_vm, q_ct):
    """Return the canonical backlog vector key: sorted totals per VM (feeder+containers), nonincreasing tuple."""
    totals = [int(q_vm[i] + sum(q_ct[i])) for i in range(N)]
    return tuple(sorted(totals, reverse=True))

def is_majorized_sorted(x_desc, y_desc):
    """x_desc, y_desc are tuples sorted nonincreasing with equal sums; return True iff x ≺ y."""
    if sum(x_desc) != sum(y_desc): 
        return False
    cx = np.cumsum(x_desc)
    cy = np.cumsum(y_desc)
    return bool(np.all(cx[:-1] <= cy[:-1]))

# -----------------------------
# Local state space (single VM)
# -----------------------------
states = []
for qv in range(C_VM + 1):
    for busy_vec in np.ndindex(*( (C_CT + 1,) * K )):
        states.append((qv,) + busy_vec)
S2I = {s: i for i, s in enumerate(states)}
nS = len(states)

def accept_feasible(x):
    qv = x[0]
    busy = x[1:]
    return (qv < C_VM) or any(b < C_CT for b in busy)

# -------------------------------------------------
# Local MDP (offer-rate model, ACCEPT cost = hold + θ)
# -------------------------------------------------
def build_local_mdp(theta, offer_rate=OFFER_RATE):
    """
    Returns:
      P_acc, P_blk : (nS x nS) one-step kernels under ACCEPT/BLOCK at offers
      C_acc, C_blk : (nS,) one-step costs under ACCEPT/BLOCK
    Uniformization uses Λ_loc = offer_rate + MU_VM + K*MU_CT (constant).
    Infeasible ACCEPT is treated exactly like BLOCK (same next state, cost = hold + C_BLOCK).
    """
    Lambda = offer_rate + MU_VM + K * MU_CT

    P_acc = np.zeros((nS, nS), dtype=float)
    P_blk = np.zeros((nS, nS), dtype=float)
    C_acc = np.zeros(nS, dtype=float)
    C_blk = np.zeros(nS, dtype=float)

    for i, s in enumerate(states):
        qv, *busy = s
        c_busy = sum(busy)
        hold = H_VM * qv + H_CT * c_busy

        # Arrival-offer transition (rate = offer_rate)
        C_blk[i] = hold 
        P_blk[i, S2I[s]] += offer_rate / Lambda

        if accept_feasible(s):
            C_acc[i] = hold + theta
            if qv < C_VM:
                s_acc = (qv + 1, *busy)
            else:
                nb = list(busy)
                for k in range(K):
                    if nb[k] < C_CT:
                        nb[k] += 1
                        break
                s_acc = (qv, *nb)
        else:
            C_acc[i] = hold  # infeasible accept == block
            s_acc = s
        P_acc[i, S2I[s_acc]] += offer_rate / Lambda

        # Promotion (rate = MU_VM)
        if (qv > 0) and (c_busy < K):
            nb = list(busy)
            for k in range(K):
                if nb[k] < C_CT:
                    nb[k] += 1
                    break
            s_prom = (qv - 1, *nb)
        else:
            s_prom = s
        P_acc[i, S2I[s_prom]] += MU_VM / Lambda
        P_blk[i, S2I[s_prom]] += MU_VM / Lambda

        # Container services (each rate = MU_CT)
        for k in range(K):
            if busy[k] > 0:
                nb = list(busy)
                nb[k] -= 1
                s_srv = (qv, *nb)
            else:
                s_srv = s
            P_acc[i, S2I[s_srv]] += MU_CT / Lambda
            P_blk[i, S2I[s_srv]] += MU_CT / Lambda

    return P_acc, P_blk, C_acc, C_blk

# -------------------------------------------------
# Policy iteration for the local MDP at a given θ
# -------------------------------------------------
def policy_iteration_local(theta, lam_cluster=None, offer_rate=OFFER_RATE, max_iter=80, tol=1e-10):
    P_acc, P_blk, C_acc, C_blk = build_local_mdp(theta, offer_rate)
    policy = np.ones(nS, dtype=int)  # initialize to ACCEPT everywhere

    I = np.eye(nS)
    for _ in range(max_iter):
        P_pi = np.where(policy[:, None] == 1, P_acc, P_blk)
        C_pi = np.where(policy == 1, C_acc, C_blk)
        V = np.linalg.solve(I - BETA * P_pi, C_pi)

        Q_acc = C_acc + BETA * (P_acc @ V)
        Q_blk = C_blk + BETA * (P_blk @ V)
        new_policy = (Q_acc < Q_blk).astype(int)

        if np.array_equal(new_policy, policy):
            break
        if np.max(np.abs(new_policy - policy)) == 0 and np.max(np.abs(Q_acc - Q_blk)) < tol:
            policy = new_policy
            break
        policy = new_policy

    # Stationary distribution under policy
    P_pi = np.where(policy[:, None] == 1, P_acc, P_blk)
    A = P_pi.T - I
    A[-1, :] = 1.0
    b = np.zeros(nS); b[-1] = 1.0
    pi_ss = np.linalg.solve(A, b)

    return V, policy, pi_ss, (P_acc, P_blk, C_acc, C_blk)

# -------------------------------------------------
# Average-rate metrics & dual lower bound
# -------------------------------------------------
def per_vm_rates_and_holding(policy, pi_ss, offer_rate=OFFER_RATE):
    E_qv  = float(sum(pi_ss[i] * states[i][0] for i in range(nS)))
    E_qct = float(sum(pi_ss[i] * sum(states[i][1:]) for i in range(nS)))
    holding_rate = H_VM * E_qv + H_CT * E_qct

    feas = np.array([accept_feasible(states[i]) for i in range(nS)], dtype=float)
    A = np.array([(policy[i] == 1) and (feas[i] == 1.0) for i in range(nS)], dtype=float)
    B = np.array([(policy[i] == 0) or ((policy[i] == 1) and (feas[i] == 0.0)) for i in range(nS)], dtype=float)

    accept_rate_i = offer_rate * float(pi_ss @ A)
    block_rate_i  = offer_rate * float(pi_ss @ B)
    return E_qv, E_qct, holding_rate, accept_rate_i, block_rate_i

def dual_lower_bound(theta, lam_cluster, policy, pi_ss, offer_rate):
    E_qv, E_qct, holding_rate, accept_rate_i, block_rate_i = \
        per_vm_rates_and_holding(policy, pi_ss, offer_rate)
    J_local_theta = holding_rate + theta*accept_rate_i
    D_theta = N * J_local_theta - theta * lam_cluster
    return D_theta, {
        "E_qvm": E_qv, "E_qct": E_qct, "holding_rate_local": holding_rate,
        "accept_rate_local": accept_rate_i, "block_rate_local": block_rate_i,
        "J_local_theta": J_local_theta, "D_theta": D_theta
    }

# -------------------------------------------------
# Solve θ* by bisection (α decreasing, allow θ<0)
# -------------------------------------------------
def dual_ascent(lam_cluster,
                tol_rate=1e-3,
                tol_theta=1e-8,
                theta_lo_init=None,
                theta_hi_init=None,
                max_iter=80,
                offer_rate_init=5.0):
    target = lam_cluster / N

    # 4.a) pick an offer rate that makes target reachable
    #r = autotune_offer_rate(lam_cluster, offer_rate_init=offer_rate_init, safety=1.15)
    r=lam_cluster / N

    # helper: solve local and get per-VM accept rate at this theta and rate r
    def solve_and_alpha(th):
        V, policy, pi_ss, parts = policy_iteration_local(th, offer_rate=r)
        _, _, _, alpha_i, _ = per_vm_rates_and_holding(policy, pi_ss, r)
        return alpha_i, (V, policy, pi_ss, parts)

    # 4.b) fixed bracket on [-C_BLOCK, +C_BLOCK]
    lo = -C_BLOCK if theta_lo_init is None else float(theta_lo_init)
    hi = +C_BLOCK if theta_hi_init is None else float(theta_hi_init)

    a_lo, pack_lo = solve_and_alpha(lo)   # α at very negative θ (maximal)
    a_hi, pack_hi = solve_and_alpha(hi)   # α at very positive θ (minimal)
    
    if a_lo < target - tol_rate:
    # KKT boundary: optimal at θ = -C_BLOCK
        return r, lo, *pack_lo

    # If α(hi) is already above target (rare), extend hi until below.
    tries = 0
    while a_hi > target + tol_rate and tries < 6:
        hi = min(hi + 0.5*C_BLOCK, 5*C_BLOCK)
        a_hi, pack_hi = solve_and_alpha(hi)
        tries += 1

    # Ensure bracket α(lo) >= target >= α(hi). If not, just return the closest side.
    if not (a_lo >= target and a_hi <= target):
        if abs(a_lo - target) <= abs(a_hi - target):
            return r, lo, *pack_lo
        else:
            return r, hi, *pack_hi

    # 4.c) Bisection on a decreasing α(θ)
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        a_mid, pack_mid = solve_and_alpha(mid)
        if abs(a_mid - target) <= tol_rate:
            return r, mid, *pack_mid
        if a_mid > target:
            lo, a_lo, pack_lo = mid, a_mid, pack_mid
        else:
            hi, a_hi, pack_hi = mid, a_mid, pack_mid # keep the pack for hi

    # --- after your bisection loop, you have: lo,a_lo,pack_lo and hi,a_hi,pack_hi
    # make sure you have the last mid, a_mid, pack_mid from the final iteration

    def eval_D(theta, pack):
        V, policy, pi_ss, parts = pack
        D_theta, _ = dual_lower_bound(theta, lam_cluster, policy, pi_ss, offer_rate=r)
        return D_theta

    # Evaluate candidates
    cands = []
    cands.append( (eval_D(lo, pack_lo),  lo,  pack_lo) )
    cands.append( (eval_D(hi, pack_hi),  hi,  pack_hi) )
    if 'pack_mid' in locals():  # if you have mid from the loop
        cands.append( (eval_D(mid, pack_mid), mid, pack_mid) )

    # Pick the θ with the largest D(θ)
    cands.sort(key=lambda t: t[0], reverse=True)
    D_best, theta_best, pack_best = cands[0]
    V_best, policy_best, pi_ss_best, parts_best = pack_best
    return r, theta_best, V_best, policy_best, pi_ss_best, parts_best


# -------------------------------------------------
# Advantage table for routing (policy-aware)
# -------------------------------------------------
def advantage_table(V, parts, policy):
    P_acc, P_blk, C_acc, C_blk = parts
    Q_acc = C_acc + BETA * (P_acc @ V)
    Q_blk = C_blk + BETA * (P_blk @ V)
    adv   = Q_acc - Q_blk
    for s in range(nS):
        if (not accept_feasible(states[s])) or (policy[s] == 0):
            adv[s] = inf
    return adv
