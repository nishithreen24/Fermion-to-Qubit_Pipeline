import numpy as np
import matplotlib.pyplot as plt
# Load
data = np.load('/Users/nishithreen/siam_results.npz', allow_pickle=True)

# Extract the assembled dict — it's wrapped in a 0-d object array
assembled = data['assembled'].item()   # .item() unwraps the 0-d array

# See what's in it
print(assembled.keys())
# → dict_keys: (N_bath, r) tuples like (1,2), (1,3), (2,2) etc.

# Access one entry
sw = assembled[(1, 2)]   # N_bath=1, r=2
print(sw.keys())
# → U_values, E0_ed, E0_xbk, D_occ, n_imp, t_wall, evals_list, N_bath, r

# Example: print ground energies for N_bath=1, r=2
print("U values:", sw['U_values'])
print("E0 (ED):", sw['E0_ed'])
print("E0 (XBK):", sw['E0_xbk'])
print("Double occupancy:", sw['D_occ'])

plt.plot(sw['U_values'], sw['E0_ed'], label='ED')
plt.plot(sw['U_values'], sw['E0_xbk'], label='XBK')
plt.legend()
plt.show()