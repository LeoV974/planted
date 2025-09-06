import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def sample_graph_k2(n, d):
    L = list(range(n))
    R = list(range(n, 2*n))
    planted = {(i, i+n) for i in L}
    p = d / n
    noise = set()
    for i in L:
        for j in R:
            if j != i+n and random.random() < p:
                noise.add((i, j))
    return planted, noise, L, R

def build_conflict_graph_k2(edge_list):
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

def sample_random_matching_k2(edge_list, conflicts, vertex_to_edges, L):
    matching = []
    used_edges = set()
    used_vertices = set()
    for u in L:
        candidates = [i for i in vertex_to_edges[u]
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

def monte_carlo_count_matchings_k2(n, d, num_samples=10000):
    planted, noise, L, R = sample_graph_k2(n, d)
    all_edges = list(planted | noise)
    if len(all_edges) < n:
        return 0, planted, noise, []

    conflicts, vertex_to_edges = build_conflict_graph_k2(all_edges)
    
    # Keep all sampled matchings, including duplicates and planted
    sampled_matchings = []
    for _ in range(num_samples):
        matching = sample_random_matching_k2(all_edges, conflicts, vertex_to_edges, L)
        if matching is not None:
            sampled_matchings.append(matching)

    # Estimate total using duplicates
    unique_matchings = set(sampled_matchings)
    num_unique = len(unique_matchings)
    success_rate = len(sampled_matchings) / num_samples if num_samples > 0 else 0
    estimated_total = int(num_unique / success_rate) if success_rate > 0 else 0

    return estimated_total, planted, noise, sampled_matchings

def analyze_matchings_vs_d_k2(n=10, d_values=None, num_samples=50000, num_trials=500):
    if d_values is None:
        d_values = [0.2, 0.5, 0.75, 0.9, 1, 1.5, 2, 3]
    results = []
    print(f"Analyzing n={n} vertices per side...")
    print(f"{'d':<5} {'Avg':<12} {'Std':<12} {'Min':<8} {'Max':<8}")
    print("-" * 50)
    for d in d_values:
        counts = []
        for _ in range(num_trials):
            count, _, _, _ = monte_carlo_count_matchings_k2(n, d, num_samples)
            counts.append(count)
        avg_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        results.append((d, avg_count, std_count, min_count, max_count))
        print(f"{d:<5.1f} {avg_count:<12.1f} {std_count:<12.1f} {min_count:<8d} {max_count:<8d}")
    return results

def plot_results(results):
    d_vals = [r[0] for r in results]
    avg_counts = [r[1] for r in results]
    std_devs = [r[2] for r in results]
    plt.figure(figsize=(10, 6))
    plt.errorbar(d_vals, avg_counts, yerr=std_devs, marker='o', capsize=5)
    plt.xlabel('d (noise parameter)')
    plt.ylabel('Average number of perfect matchings')
    plt.title('Perfect Matchings vs Noise Parameter d (k=2)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    results = analyze_matchings_vs_d_k2()
    plot_results(results)
