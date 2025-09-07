import sys, json, os
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
k = int(sys.argv[2])

with open(path, "r") as f:
    data = json.load(f)

# d*(k) values
special_d_dict = {2: 1, 3: 6.1766, 4: 18.995}

d_vals, avg, std = [], [], []
for row in data:
    d_vals.append(float(row[0]))
    avg.append(float(row[1]))
    std.append(float(row[2]))

order = np.argsort(d_vals)
d = np.array(d_vals)[order]
a = np.array(avg)[order]
s = np.array(std)[order]

plt.figure(figsize=(8,5))
plt.errorbar(d, a, yerr=s, marker='o', capsize=5)
plt.axvline(special_d_dict[k], color='red', linestyle='--', label=f'd*({k})')
plt.xlabel('d (noise parameter)')
plt.ylabel('Average number of distinct perfect matchings (per trial)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
