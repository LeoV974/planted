import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json

def sample_graph_k3(n, d):
    L = list(range(n))
    M = list(range(n, 2*n))
    R = list(range(2*n, 3*n))
    planted = {(i, i+n, i+2*n) for i in L}
    p = d / (n * n)
    noise = set()
    for l in L:
        for m in M:
            for r in R:
                if (l, m, r) not in planted and random.random() < p:
                    noise.add((l, m, r))
    return planted, noise, L, M, R

def build_conflict_graph(edge_list):
    conflicts = [set() for _ in range(len(edge_list))]
    vertex_to_edges = defaultdict(list)
    for i, edge in enumerate(edge_list):
        for v in edge:
            vertex_to_edges[v].append(i)
    for edges in vertex_to_edges.values():
        for i in edges:
            for j in edges:
                if i != j:
                    conflicts[i].add(j)
    return conflicts, vertex_to_edges

def sample_random_matching_k3(edge_list, conflicts, vertex_to_edges, L):
    matching = []
    used_edges = set()
    used_vertices = set()
    for l_vertex in L:
        candidates = [i for i in vertex_to_edges[l_vertex]
                      if i not in used_edges and all(v not in used_vertices for v in edge_list[i])]
        if candidates:
            chosen = random.choice(candidates)
            chosen_edge = edge_list[chosen]
            matching.append(chosen_edge)
            used_edges.add(chosen)
            used_vertices.update(chosen_edge)
            used_edges.update(conflicts[chosen])
    if len(matching) == len(L):
        return frozenset(matching)
    return None

def monte_carlo_count_matchings(n, d, num_samples=10000):
    planted, noise, L, M, R = sample_graph_k3(n, d)
    all_edges = list(planted | noise)
    if len(all_edges) < n:
        return 0, planted, noise, set()

    conflicts, vertex_to_edges = build_conflict_graph(all_edges)
    found_matchings = set()
    for _ in range(num_samples):
        matching = sample_random_matching_k3(all_edges, conflicts, vertex_to_edges, L)
        if matching is not None:
            found_matchings.add(matching)

    return len(found_matchings), planted, noise, found_matchings

def analyze_matchings_vs_d(n=10, d_values=None, num_samples=10000, num_trials=100):
    if d_values is None:
        d_values = [4, 5, 6, 7, 8]

    results = []
    print(f"Analyzing n={n} vertices per side...")
    print(f"{'d':<5} {'Avg':<12} {'Std':<12} {'Min':<8} {'Max':<8}")
    print("-" * 50)

    for d in d_values:
        counts = []
        for _ in range(num_trials):
            count, _, _, _ = monte_carlo_count_matchings(n, d, num_samples)
            counts.append(count)
        avg_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        results.append((d, float(avg_count), float(std_count), int(min_count), int(max_count)))
        print(f"{d:<5.1f} {avg_count:<12.1f} {std_count:<12.1f} {min_count:<8d} {max_count:<8d}")

    # Save results to JSON
    with open("results_k3.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def plot_results(results):
    d_vals = [r[0] for r in results]
    avg_counts = [r[1] for r in results]
    std_devs = [r[2] for r in results]
    plt.figure(figsize=(10, 6))
    plt.errorbar(d_vals, avg_counts, yerr=std_devs, marker='o', capsize=5)
    plt.xlabel('d (noise parameter)')
    plt.ylabel('Average number of distinct perfect matchings (per trial)')
    plt.title('Distinct Perfect Matchings vs Noise Parameter d (k=3)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = analyze_matchings_vs_d()
    plot_results(results)
