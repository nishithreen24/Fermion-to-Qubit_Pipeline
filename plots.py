import matplotlib.pyplot as plt
import numpy as np
from runner import summary
V = summary['V_values']
V_c = 2.0   # known transition for t-V model

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Programme 1
axes[0,0].plot(V, summary['spectral_gap_LH'], 'o-')
axes[0,0].axvline(V_c, color='r', linestyle='--', label='V/t=2')
axes[0,0].set_title('Spectral gap λ₂(L_H)')
axes[0,0].set_xlabel('V'); axes[0,0].legend()

axes[0,1].plot(V, summary['mean_kappa'], 'o-')
axes[0,1].axvline(V_c, color='r', linestyle='--')
axes[0,1].set_title('Mean Ollivier-Ricci curvature')
axes[0,1].set_xlabel('V')

axes[0,2].plot(V, summary['std_kappa'], 'o-')
axes[0,2].axvline(V_c, color='r', linestyle='--')
axes[0,2].set_title('Std of curvature')
axes[0,2].set_xlabel('V')

# Programme 2
axes[1,0].plot(V, summary['ground_energy'], 'o-')
axes[1,0].axvline(V_c, color='r', linestyle='--')
axes[1,0].set_title('Ground state energy')
axes[1,0].set_xlabel('V')

axes[1,1].plot(V, summary['wasserstein_ref'], 'o-')
axes[1,1].axvline(V_c, color='r', linestyle='--')
axes[1,1].set_title('W(μ_V, μ_ref) Sinkhorn')
axes[1,1].set_xlabel('V')

axes[1,2].plot(V, summary['spectral_gap_metro'], 'o-')
axes[1,2].axvline(V_c, color='r', linestyle='--')
axes[1,2].set_title('Metropolis spectral gap')
axes[1,2].set_xlabel('V')

plt.tight_layout()
plt.savefig('geometry_sweep.png', dpi=150)
plt.show()