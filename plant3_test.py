import random
from ortools.sat.python import cp_model
import time
import json

def generate_instance(n, d, seed=0):
    rng = random.Random(seed)
    U = list(range(n)); V = list(range(n)); W = list(range(n))
    planted = {(i, i, i) for i in range(n)}
    m = int(d*n)
    noise = set()
    while len(noise) < m:
        e = (rng.randrange(n), rng.randrange(n), rng.randrange(n))
        if e not in planted:
            noise.add(e)
    E = planted | noise
    return U, V, W, planted, E

def find_alternate_matching(U, V, W, planted, E, time_limit=10.0):
    model = cp_model.CpModel()
    x = {e: model.NewBoolVar(f"x_{e[0]}_{e[1]}_{e[2]}") for e in E}
    for i in U:
        model.Add(sum(x[e] for e in E if e[0] == i) == 1)
    for j in V:
        model.Add(sum(x[e] for e in E if e[1] == j) == 1)
    for k in W:
        model.Add(sum(x[e] for e in E if e[2] == k) == 1)
    model.Add(sum(x[e] for e in planted) <= len(U) - 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        M = {e for e in E if solver.Value(x[e]) == 1}
        t = len(M - planted)
        return True, M, t
    return False, None, None

def sweep(n=100, ds=(2, 4, 6, 8, 10, 12), trials=20, seed=0):
    rng = random.Random(seed)
    print(f"--- Running sweep with n={n}, k=3 ---")
    results = []
    for d in ds:
        start_time_d_block = time.time()
        hits, t_stats = 0, []
        print(f"\n--- Sweeping d = {d} ---")
        for r in range(trials):
            start_time_trial = time.time()
            trial_seed = rng.randrange(10**9)
            U, V, W, planted, E = generate_instance(n, d, seed=trial_seed)
            ok, M, t = find_alternate_matching(U, V, W, planted, E, time_limit=90)
            elapsed_trial = time.time() - start_time_trial
            if ok:
                hits += 1
                t_stats.append(t)
                print(f"  d={d}, trial {r+1:2}/{trials}: SUCCESS (t={t:2}) in {elapsed_trial:4.1f}s")
            else:
                print(f"  d={d}, trial {r+1:2}/{trials}: FAILED         in {elapsed_trial:4.1f}s")
        rate = hits / trials
        avg_t = sum(t_stats)/len(t_stats) if t_stats else 0.0
        elapsed_d_block = time.time() - start_time_d_block
        print(f"SUMMARY d={d:>3}  success_rate={rate:5.2f}  avg_t|success={avg_t:5.2f}  (total_time={elapsed_d_block:4.1f}s)")
        results.append((d, rate, avg_t))
    return results

if __name__ == "__main__":
    results = sweep(n=100, ds=[5.5, 6, 6.5, 10], trials=10)
    success_rates = {str(d): rate for d, rate, _ in results}
    with open("success_rates.json", "w") as f:
        json.dump(success_rates, f, indent=2)
    print("\nSaved success rates to success_rates.json")
