import sys, json, os
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
with open(path, "r") as f:
    data = json.load(f)

d_vals, avg, std = [], [], []
for row in data:
    d_vals.append(float(row[0])); avg.append(float(row[1])); std.append(float(row[2]))
order = np.argsort(d_vals)
d = np.array(d_vals)[order]
a = np.array(avg)[order]
s = np.array(std)[order]

plt.figure(figsize=(8,5))
plt.errorbar(d, a, yerr=s, marker='o', capsize=5)
plt.xlabel('d (noise parameter)')
plt.ylabel('Average number of distinct perfect matchings (per trial)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
