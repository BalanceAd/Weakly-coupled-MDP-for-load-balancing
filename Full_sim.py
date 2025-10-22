import MDP_scaled.py


# -------------------------------------------------
# Global CTMC simulation using the assembled policy
# -------------------------------------------------
def simulate_global(
    lam_cluster,
    adv_vec,
    policy,
    horizon=60000.0,
    warm=10000.0,
    seed=1,
    # TD(0) params
    rho=0.01,                # continuous-time discount rate (>0); gamma = e^{-rho * dt}
    alpha0=0.2,              # initial stepsize
    alpha_cool=1000.0,       # stepsize schedule α_s = α0 / (1 + visits[s]/alpha_cool)
    min_visits_for_eval=10,  # only trust states seen >= this many times
    # verification sampling
    max_pairs_per_S=50       # how many (x≺y) pairs per total S to test at the end
):
    rng = random.Random(seed)

    # State
    q_vm = [0]*N
    q_ct = [[0]*K for _ in range(N)]

    # Metrics
    t = 0.0
    last_t = 0.0
    area_L = 0.0
    arrived = 0
    blocked = 0

    # Value tables
    V = defaultdict(float)      # V[key]
    visits = defaultdict(int)   # visit counts for stepsize schedule

    # Initialize current state key
    s_key = canonical_key(q_vm, q_ct)

    while t < horizon:
        # total rate (arrivals + eligible promotions + services)
        rate = lam_cluster
        for i in range(N):
            if q_vm[i] > 0 and sum(q_ct[i]) < K:
                rate += MU_VM
        for i in range(N):
            for k in range(K):
                if q_ct[i][k] > 0:
                    rate += MU_CT
        if rate <= 0.0:
            break

        dt = rng.expovariate(rate)
        t_next = t + dt

        # instantaneous holding (constant between events)
        Ltot = H_VM*sum(q_vm) + H_CT*sum(sum(row) for row in q_ct)

        # discounted reward over (t, t+dt): ∫ e^{-ρ(τ-t)} L dτ = L * (1 - e^{-ρ dt}) / ρ
        # block penalty (if any) will be added at event time with an extra e^{-ρ dt} factor
        disc_int = (1.0 - np.exp(-rho*dt)) / max(rho, 1e-12)
        R_cont = Ltot * disc_int

        # Sample event
        rpick = rng.random() * rate
        event = None
        if rpick < lam_cluster:
            event = "arrival"
        else:
            rpick -= lam_cluster
            # promotions
            for i in range(N):
                if q_vm[i] > 0 and sum(q_ct[i]) < K:
                    if rpick < MU_VM:
                        # execute promotion at end of interval
                        event = ("promote", i)
                        break
                    rpick -= MU_VM
            if event is None:
                for i in range(N):
                    for k in range(K):
                        if q_ct[i][k] > 0:
                            if rpick < MU_CT:
                                event = ("service", i, k)
                                break
                            rpick -= MU_CT
                    if event is not None:
                        break

        # Compute next state by simulating the event
        # We'll also tally steady-state metrics after warm-up
        # TD target needs s' and an impulse reward if block occurs
        # Copy references
        q_vm2 = q_vm
        q_ct2 = q_ct

        # Discount factor to next state
        gamma = float(np.exp(-rho*dt))
        R_impulse = 0.0

        # Apply event at t_next
        if event == "arrival":
            if t_next > warm:
                arrived += 1
            # choose VM among feasible+locally-ACCEPTING with min advantage
            best_i, best_adv = None, float("inf")
            for i in range(N):
                s = (q_vm[i], *q_ct[i])
                if accept_feasible(s):
                    s_idx = S2I[s]
                    if policy[s_idx] == 1:
                        a = adv_vec[s_idx]
                        if a < best_adv:
                            best_adv, best_i = a, i
            if best_i is None:
                if t_next > warm:
                    blocked += 1
                # impulse block penalty at the boundary, discounted by gamma
                R_impulse += C_BLOCK
            else:
                if q_vm[best_i] < C_VM:
                    q_vm[best_i] += 1
                else:
                    placed = False
                    for k in range(K):
                        if q_ct[best_i][k] < C_CT:
                            q_ct[best_i][k] += 1
                            placed = True
                            break
                    if not placed:
                        if t_next > warm:
                            blocked += 1
                        R_impulse += C_BLOCK

        elif event and event[0] == "promote":
            i = event[1]
            if q_vm[i] > 0 and sum(q_ct[i]) < K:
                for k in range(K):
                    if q_ct[i][k] < C_CT:
                        q_vm[i] -= 1
                        q_ct[i][k] += 1
                        break

        elif event and event[0] == "service":
            i, k = event[1], event[2]
            if q_ct[i][k] > 0:
                q_ct[i][k] -= 1

        # TD(0) update for the canonical key observed over (t, t+dt]
        s_next_key = canonical_key(q_vm, q_ct)
        # Total discounted one-step reward = continuous part + discounted impulse at boundary
        R_total = R_cont + gamma * R_impulse
        # Bootstrap target
        target = R_total + gamma * V[s_next_key]
        # Stepsize schedule
        a = alpha0 / (1.0 + visits[s_key] / max(alpha_cool, 1e-9))
        V[s_key] += a * (target - V[s_key])
        visits[s_key] += 1

        # Advance time & accumulate steady-state area after warm-up
        if t_next > warm:
            area_L += Ltot * (t_next - max(t, warm))
        t = t_next
        last_t = t_next
        s_key = s_next_key

    # Steady-state metrics
    duration = max(horizon - warm, 1e-12)
    Lbar = area_L / duration
    block_rate = blocked / duration
    cost_rate = Lbar + C_BLOCK * block_rate

    # ---- Build Schur-convexity verification from learned V ----
    # Group visited states by total S and sample pairs (x≺y)
    by_S = {}
    for key, v in V.items():
        if visits[key] >= min_visits_for_eval:
            S = sum(key)
            by_S.setdefault(S, []).append((key, v))

    rows = []
    for S, lst in by_S.items():
        if len(lst) < 2:
            continue
        rng.shuffle(lst)
        # Construct up to max_pairs_per_S pairs
        tested = 0
        for i in range(len(lst)):
            if tested >= max_pairs_per_S: break
            xi, Vi = lst[i]
            for j in range(i+1, len(lst)):
                yj, Vj = lst[j]
                # check majorization on sorted keys (they already are)
                # We want xi ≺ yj (xi more balanced => smaller value)
                if is_majorized_sorted(xi, yj):
                    holds = (Vi <= Vj + 1e-8)
                    rows.append(dict(S=S, x=xi, y=yj, Vx=Vi, Vy=Vj, holds=holds))
                    tested += 1
                    break  # move to next i

    detail = pd.DataFrame(rows)
    if not detail.empty:
        summary = (detail.groupby("S", as_index=False)
                         .agg(violations=("holds", lambda s: int((~s).sum())),
                              total=("holds", "size"))
                         .assign(violation_rate=lambda d: d["violations"]/d["total"]))
    else:
        summary = pd.DataFrame(columns=["S","violations","total","violation_rate"])

    perf = {"L": Lbar, "block_rate": block_rate, "arrivals": arrived, "duration": duration}
    return cost_rate, perf, V, visits, detail, summary