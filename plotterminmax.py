import sys, json, os
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
k = int(sys.argv[2])

with open(path, "r") as f:
    data = json.load(f)

special_d_dict = {2: 1, 3: 6.1766, 4: 18.995}

d_vals, avg, std, min_vals, max_vals = [], [], [], [], []
for row in data:
    d_vals.append(float(row[0]))
    avg.append(float(row[1]))
    std.append(float(row[2]))
    min_vals.append(float(row[3]))
    max_vals.append(float(row[4]))

order = np.argsort(d_vals)
d = np.array(d_vals)[order]
a = np.array(avg)[order]
s = np.array(std)[order]
mn = np.array(min_vals)[order]
mx = np.array(max_vals)[order]

plt.figure(figsize=(8,5))
plt.plot(d, a, marker='o')
plt.fill_between(d, mn, mx, color='gray', alpha=0.2, label='min–max')
plt.axvline(special_d_dict[k], color='red', linestyle='--', label=f'd*({k})')
plt.xlabel('d (noise parameter)')
plt.ylabel('Number of distinct perfect matchings')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
