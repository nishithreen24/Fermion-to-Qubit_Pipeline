from Models import *
from playground import *
from SIAM import *

H = build_siam_hamiltonian(N=2)

Q = bravyi_kitaev(H)

E1 = exact_ground_energy(Q)

Qdiag = XBK_fast_correct(Q, n=of.count_qubits(Q), r=2, p=0)

E2 = exact_ground_energy(Qdiag)

print(E1, E2)