import argparse, random
import numpy as np
import pandas as pd
# direct import 
import MDP_scaled.py as MDP



def cap_proximity_after_assign(MDP, i, q_vm, q_ct):
    """Cheap proxy for block risk after placing on VM i: (used+1)/cap in [0,1]."""
    used = q_vm[i] + sum(q_ct[i])
    cap  = MDP.C_VM + MDP.K * MDP.C_CT
    return min(1.0, (used + 1) / max(cap, 1))

def promotion_slack(MDP, i, q_vm, q_ct):
    """Tie-breaker: how many immediate promotions could happen soon on VM i."""
    # count free container slots + whether there is something to promote
    free_ct_slots = sum(1 for k in range(MDP.K) if q_ct[i][k] < MDP.C_CT)
    has_feeder    = 1 if q_vm[i] > 0 else 0
    return free_ct_slots + has_feeder

def jsq_feasible_vm(q_vm, q_ct):
    best_i, best_load = None, float("inf")
    for i in range(MDP.N):
        s = (q_vm[i], *q_ct[i])
        if MDP.accept_feasible(s):
            load_i = q_vm[i] + sum(q_ct[i])
            if load_i < best_load:
                best_i, best_load = i, load_i
    return best_i

class RRPointer:
    def __init__(self): self.ptr = 0
    def next_feasible(self, q_vm, q_ct):
        for _ in range(MDP.N):
            i = self.ptr
            self.ptr = (self.ptr + 1) % MDP.N
            s = (q_vm[i], *q_ct[i])
            if MDP.accept_feasible(s):
                return i
        return None

def fcma_style_vm(q_vm, q_ct):
    # pack idle containers first, else feeder with most free slots
    best_i, best_idle = None, -1
    for i in range(MDP.N):
        idle_ct = sum(1 for k in range(MDP.K) if q_ct[i][k] < MDP.C_CT)
        if idle_ct > 0 and idle_ct > best_idle:
            best_idle, best_i = idle_ct, i
    if best_i is not None:
        return best_i
    best_i, best_space = None, -1
    for i in range(MDP.N):
        space = MDP.C_VM - q_vm[i]
        if space > 0 and space > best_space:
            best_space, best_i = space, i
    return best_i

def simulate(lam_cluster, mode, adv_vec=None, policy=None, tau_low=0.30,
             horizon=30000.0, warm=5000.0, seed=1, eta=0.0, eps=0.0,
             tau_down=0.30, tau_up=0.35):
    rng = random.Random(seed)
    q_vm = [0]*MDP.N
    q_ct = [[0]*MDP.K for _ in range(MDP.N)]
    rr = RRPointer()

    # Mode state for hysteresis
    mode_state = {"phase": "LOW"}  # start LOW; we’ll switch up/down by hysteresis thresholds

    t = 0.0; area_L = 0.0; arrived = 0; blocked = 0
    while t < horizon:
        rate = lam_cluster
        for i in range(MDP.N):
            if q_vm[i] > 0 and sum(q_ct[i]) < MDP.K: rate += MDP.MU_VM
        for i in range(MDP.N):
            for k in range(MDP.K):
                if q_ct[i][k] > 0: rate += MDP.MU_CT
        if rate <= 0: break

        dt = rng.expovariate(rate); t_next = t + dt
        Ltot = MDP.H_VM*sum(q_vm) + MDP.H_CT*sum(sum(r) for r in q_ct)
        if t_next > warm: area_L += Ltot * (t_next - max(t, warm))

        rpick = rng.random()*rate
        if rpick < lam_cluster:  # ARRIVAL
            if t_next > warm: arrived += 1
            def place(idx):
                nonlocal blocked
                if idx is None:
                    if t_next > warm: blocked += 1
                    return
                if q_vm[idx] < MDP.C_VM: q_vm[idx] += 1
                else:
                    for kk in range(MDP.K):
                        if q_ct[idx][kk] < MDP.C_CT:
                            q_ct[idx][kk] += 1; break
                    else:
                        if t_next > warm: blocked += 1

            if mode == "jsq":
                place(jsq_feasible_vm(q_vm, q_ct))
            elif mode == "rr":
                place(rr.next_feasible(q_vm, q_ct))
            elif mode == "fcma":
                place(fcma_style_vm(q_vm, q_ct))
            elif mode == "load_aware":
                assert adv_vec is not None and policy is not None, "Need adv_vec/policy for load_aware."

                S   = sum(q_vm) + sum(sum(row) for row in q_ct)
                CAP = MDP.N * (MDP.C_VM + MDP.K * MDP.C_CT)

                # Hysteresis phase update
                if mode_state["phase"] == "LOW" and S >= tau_up * CAP:
                    mode_state["phase"] = "HIGH"
                elif mode_state["phase"] == "HIGH" and S <= tau_down * CAP:
                    mode_state["phase"] = "LOW"

                if mode_state["phase"] == "LOW":
                    # JSQ among feasible
                    best_i = jsq_feasible_vm(q_vm, q_ct)
                    # If no feasible VM, then it is a true block
                    place_on_vm(best_i)
                else:
                    # HIGH load: score candidates with blocking-aware proxy + promotion slack tie-break
                    C = []
                    for i in range(MDP.N):
                        s = (q_vm[i], *q_ct[i])
                        if MDP.accept_feasible(s):
                            s_idx = MDP.S2I[s]
                            adv   = adv_vec[s_idx]
                            prox  = cap_proximity_after_assign(MDP, i, q_vm, q_ct)  # [0,1]
                            score = adv + eta * prox  # lower is better
                            slack = promotion_slack(MDP, i, q_vm, q_ct)  # higher is better
                            C.append((score, -slack, i))  # sort by score asc, then slack desc

                    if C:
                        C.sort()
                        # epsilon-randomize among top 2 to avoid herding
                        top_k = min(2, len(C))
                        if eps > 0.0 and random.random() < eps:
                            choice = C[random.randrange(top_k)][-1]
                        else:
                            choice = C[0][-1]
                        place_on_vm(choice)
                    else:
                        # Strict “don’t block if feasible”: try JSQ among feasible as a last resort
                        best_i = jsq_feasible_vm(MDP, q_vm, q_ct)
                        place_on_vm(best_i)
            else:
                raise ValueError(mode)
        else:
            rpick -= lam_cluster
            promoted = False
            for i in range(MDP.N):
                if q_vm[i] > 0 and sum(q_ct[i]) < MDP.K:
                    if rpick < MDP.MU_VM:
                        for kk in range(MDP.K):
                            if q_ct[i][kk] < MDP.C_CT:
                                q_vm[i] -= 1; q_ct[i][kk] += 1; promoted = True; break
                        break
                    rpick -= MDP.MU_VM
            if not promoted:
                for i in range(MDP.N):
                    for k in range(MDP.K):
                        if q_ct[i][k] > 0:
                            if rpick < MDP.MU_CT:
                                q_ct[i][k] -= 1; promoted = True; break
                            rpick -= MDP.MU_CT
                    if promoted: break
        t = t_next

    dur = max(horizon - warm, 1e-12)
    Lbar = area_L/dur
    block = blocked/dur
    cost = Lbar + MDP.C_BLOCK*block
    return cost, {"L": Lbar, "block_rate": block, "arrivals": arrived, "duration": dur}

def run_compare(lams, seeds, horizon, warm, tau_low, eta, eps, tau_down, tau_up):
    rows = []
    for lam in lams:
        print(f"λ={lam}")
        r, theta_star, V, policy, pi_ss, parts = MDP.dual_ascent(lam)
        adv = MDP.advantage_table(V, parts, policy)
        for seed in seeds:
            for mode in ["load_aware","jsq","rr","fcma"]:
                cost, perf = simulate(lam, mode, adv, policy, tau_low, horizon, warm, seed, eta, eps, tau_down, tau_up) \
                             if mode=="load_aware" else \
                             simulate(lam, mode, None, None, tau_low, horizon, warm, seed, eta, eps, tau_down, tau_up)
                rows.append(dict(policy=("fcma_style" if mode=="fcma" else mode),
                                 lam=lam, seed=seed, cost_rate=cost, **perf))
    return pd.DataFrame(rows)



