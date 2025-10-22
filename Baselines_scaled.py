import math, random
import numpy as np
import pandas as pd

# ---------------------------
# Model parameters (baseline)
# ---------------------------
N        = 50   # VMs
K        = 2    # containers per VM
C_VM     = 3    # VM queue capacity
C_CT     = 1    # container capacity (binary)
MU_VM    = 2.0  # VM->container promotion rate
MU_CT    = 1.0  # container service rate
H_VM     = 1.0  # holding cost (VM queue)
H_CT     = 1.0  # holding cost (containers)
C_BLOCK  = 15.0 # blocking cost per rejected job

# ---------------------------
# Helpers
# ---------------------------
def vm_has_capacity(qv, q1, q2):
    """Feasible to accept at this VM right now?"""
    return (qv < C_VM) or (q1 < C_CT) or (q2 < C_CT)

def admit_into_vm(qv, q1, q2):
    """Return new (qv,q1,q2) after admission (VM queue preferred, then containers).
       If no capacity, return unchanged and a flag."""
    if qv < C_VM:
        return (qv+1, q1, q2), True
    if q1 < C_CT:
        return (qv, q1+1, q2), True
    if q2 < C_CT:
        return (qv, q1, q2+1), True
    return (qv, q1, q2), False

def pick_vm_jsq(qv, q1, q2):
    """JSQ over VMs that can accept now; tie-break uniformly at random."""
    feas = [i for i in range(N) if vm_has_capacity(qv[i], q1[i], q2[i])]
    if not feas:
        return None
    # total backlog b_i = q_vm + q1 + q2
    b = [qv[i] + q1[i] + q2[i] for i in feas]
    m = min(b)
    cand = [feas[i] for i,bi in enumerate(b) if bi == m]
    return random.choice(cand)

def pick_vm_rr(qv, q1, q2, rr_ptr):
    """RR pointer advances until it finds a feasible VM; if none, return None.
       Returns (index, new_rr_ptr)."""
    start = rr_ptr
    for _ in range(N):
        if vm_has_capacity(qv[rr_ptr], q1[rr_ptr], q2[rr_ptr]):
            idx = rr_ptr
            rr_ptr = (rr_ptr + 1) % N
            return idx, rr_ptr
        rr_ptr = (rr_ptr + 1) % N
        if rr_ptr == start:
            break
    return None, rr_ptr

# -----------------------------------------
# Discrete-Event Simulation (one policy)
# -----------------------------------------
def simulate_policy(lam, policy="JSQ", horizon=40000.0, warm=5000.0, seed=1):
    """
    policy ∈ {"JSQ","RR"}; baselines accept if feasible.
    """
    random.seed(seed)

    # VM state
    q_vm  = [0]*N
    q_ct1 = [0]*N
    q_ct2 = [0]*N
    rr_ptr = 0  # round-robin pointer

    t = 0.0
    last_t = 0.0
    area_L = 0.0
    area_busy_vm = [0.0]*N  # 'busy' if any container busy

    arrived = 0
    blocked = 0
    admitted = 0

    while t < horizon:
        # Build total event rate
        # Arrival
        rate = lam
        # Promotions only when VM queue>0 and some container idle
        for i in range(N):
            if q_vm[i] > 0 and (q_ct1[i] < C_CT or q_ct2[i] < C_CT):
                rate += MU_VM
        # Container services: each busy container contributes MU_CT
        for i in range(N):
            if q_ct1[i] > 0: rate += MU_CT
            if q_ct2[i] > 0: rate += MU_CT

        if rate <= 0.0:
            break

        # Next event time
        dt = random.expovariate(rate)
        t += dt

        # Statistics accumulation after warm-up
        if t > warm:
            dt_eff = t - last_t
            Ltot = sum(q_vm) + sum(q_ct1) + sum(q_ct2)
            area_L += Ltot * dt_eff
            for i in range(N):
                busy = 1.0 if (q_ct1[i] + q_ct2[i]) > 0 else 0.0
                area_busy_vm[i] += busy * dt_eff
        last_t = t

        # Select event
        r = random.random() * rate

        # --- Arrival ---
        if r < lam:
            if t > warm:
                arrived += 1
            if policy == "JSQ":
                idx = pick_vm_jsq(q_vm, q_ct1, q_ct2)
                if idx is None:
                    if t > warm: blocked += 1
                else:
                    (q_vm[idx], q_ct1[idx], q_ct2[idx]), ok = admit_into_vm(q_vm[idx], q_ct1[idx], q_ct2[idx])
                    if ok and t > warm:
                        admitted += 1
                    elif (not ok) and t > warm:
                        blocked += 1
            else:  # RR
                idx, rr_ptr = pick_vm_rr(q_vm, q_ct1, q_ct2, rr_ptr)
                if idx is None:
                    if t > warm: blocked += 1
                else:
                    (q_vm[idx], q_ct1[idx], q_ct2[idx]), ok = admit_into_vm(q_vm[idx], q_ct1[idx], q_ct2[idx])
                    if ok and t > warm:
                        admitted += 1
                    elif (not ok) and t > warm:
                        blocked += 1
            continue

        r -= lam

        # --- Promotions (scan VMs; each eligible contributes MU_VM) ---
        for i in range(N):
            eligible = (q_vm[i] > 0) and (q_ct1[i] < C_CT or q_ct2[i] < C_CT)
            if eligible:
                if r < MU_VM:
                    # promote to first idle container
                    if q_ct1[i] < C_CT:
                        q_vm[i] -= 1; q_ct1[i] += 1
                    elif q_ct2[i] < C_CT:
                        q_vm[i] -= 1; q_ct2[i] += 1
                    else:
                        # should not happen due to 'eligible' guard
                        pass
                    break
                r -= MU_VM

        # --- Container service completions (each busy contributes MU_CT) ---
        # Pass 1 for ct1, then ct2
        for i in range(N):
            if q_ct1[i] > 0:
                if r < MU_CT:
                    q_ct1[i] -= 1
                    break
                r -= MU_CT
            if q_ct2[i] > 0:
                if r < MU_CT:
                    q_ct2[i] -= 1
                    break
                r -= MU_CT

    # Final metrics
    duration    = max(horizon - warm, 1e-9)
    Lbar        = area_L / duration
    lam_eff     = admitted / duration
    W           = float('inf') if lam_eff <= 0 else (Lbar / lam_eff)
    util_mean   = np.mean([b / duration for b in area_busy_vm])
    block_prob  = 0.0 if arrived == 0 else (blocked / arrived)
    avg_cost    = (H_VM + H_CT) * 0.0  # (we use Lbar as holding cost below)
    # With h_VM=h_CT=1, holding cost rate equals expected #jobs in system:
    avg_cost    = Lbar + C_BLOCK * (blocked / duration)

    return {
        "lambda": lam,
        "policy": policy,
        "L": Lbar,
        "lambda_eff": lam_eff,
        "W": W,
        "util": util_mean,
        "block_prob": block_prob,
        "avg_cost_rate": avg_cost
    }
