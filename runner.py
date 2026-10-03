from playground import *
import numpy as np

V_values = np.linspace(0.5, 4.0, 15).tolist()
summary = run_geometry_sweep(
    L=6, t=1.0, V_values=V_values,
    beta=5.0, periodic=False,
    r=2, p=2,
    compute_metropolis=False
)