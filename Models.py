import numpy as np
from openfermion import FermionOperator


def _mode(site: int, spin: int) -> int:
    """
    Return the spin-orbital mode index for a given site and spin.

    Parameters
    ----------
    site : int
        0 = impurity, 1..N = bath sites.
    spin : int
        0 = spin-up, 1 = spin-down.

    Returns
    -------
    int
        Mode index = 2*site + spin.
    """
    return 2 * site + spin


def _number_op(site: int, spin: int) -> FermionOperator:
    """
    Return the number operator n_{site,spin} = a†_{site,spin} a_{site,spin}
    as a FermionOperator.
    """
    p = _mode(site, spin)
    return FermionOperator(((p, 1), (p, 0)))


def build_siam_hamiltonian(
    N: int,
    U: float = 4.0,
    t: float = 1.0,
    V: float = 1.0,
    eps_imp: float = 0.0,
) -> FermionOperator:
    """
    Construct the SIAM Hamiltonian for N bath sites and 1 impurity.

    Parameters
    ----------
    N : int
        Number of bath sites.  Must satisfy N >= 1 and N odd (so that there is
        always a bath site at energy eps_b = 0).
    U : float
        On-site Coulomb repulsion on the impurity.
    t : float
        Nearest-neighbour hopping of the 1D chain that the star geometry
        represents.  Sets the half-bandwidth W = 2t.
    V : float
        Impurity-chain hybridization amplitude.
    eps_imp : float
        On-site energy of the impurity level.  Zero gives particle-hole
        symmetry when combined with the particle-hole symmetric form of H_imp.

    Returns
    -------
    FermionOperator
        The full SIAM Hamiltonian as an OpenFermion FermionOperator.

    Notes
    -----
    Total number of spin-orbitals: 2*(N+1).
        Modes 0,1  -> impurity spin-up, spin-down
        Modes 2b, 2b+1 -> bath site b (b=1..N) spin-up, spin-down

    Bath parameters (from the star-geometry / 1D-chain correspondence):
        eps_b = -2t * cos(b*pi / (N+1))
        V_b   = V * sqrt(2/(N+1)) * sqrt(1 - (eps_b/(2t))^2)
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    # ------------------------------------------------------------------
    # Bath energies and hybridization amplitudes
    # ------------------------------------------------------------------
    b = np.arange(1, N + 1)                         # b = 1, ..., N
    eps_bath = -2.0 * t * np.cos(b * np.pi / (N + 1))
    V_bath   = V * np.sqrt(2.0 / (N + 1)) * np.sqrt(1.0 - (eps_bath / (2.0 * t))**2)

    H = FermionOperator()

    # ------------------------------------------------------------------
    # H_imp = U * n_{0,up} * n_{0,down}
    #         - (U/2) * (n_{0,up} + n_{0,down})
    #         + U/4
    #         + eps_imp * (n_{0,up} + n_{0,down})
    #
    # The two-body part U * n_{0,up} * n_{0,down}
    # in normal-ordered form is a†_{0,up} a_{0,up} a†_{0,dn} a_{0,dn}.
    # OpenFermion requires normal ordering, so we write it as
    # a†_{0,up} a†_{0,dn} a_{0,dn} a_{0,up}   (with sign +1 from one swap).
    # ------------------------------------------------------------------
    up_imp = _mode(0, 0)
    dn_imp = _mode(0, 1)

    # Two-body: U * a†_{up} a†_{dn} a_{dn} a_{up}
    H += FermionOperator(((up_imp, 1), (dn_imp, 1), (dn_imp, 0), (up_imp, 0)), U)

    # One-body on impurity
    one_body_coeff = -U / 2.0 + eps_imp
    for spin in range(2):
        H += FermionOperator((((_mode(0, spin), 1), (_mode(0, spin), 0))),
                             one_body_coeff)

    # Scalar
    H += FermionOperator((), U / 4.0)

    # ------------------------------------------------------------------
    # H_bath = sum_{b=1}^{N} eps_b * (n_{b,up} + n_{b,down})
    # ------------------------------------------------------------------
    for idx, (b_site, eps_b) in enumerate(zip(range(1, N + 1), eps_bath)):
        for spin in range(2):
            H += FermionOperator(
                ((_mode(b_site, spin), 1), (_mode(b_site, spin), 0)),
                eps_b,
            )

    # ------------------------------------------------------------------
    # H_hyb = sum_{b=1}^{N} V_b * sum_{spin}
    #             (a†_{imp,spin} a_{b,spin} + a†_{b,spin} a_{imp,spin})
    # ------------------------------------------------------------------
    for idx, (b_site, V_b) in enumerate(zip(range(1, N + 1), V_bath)):
        for spin in range(2):
            imp_mode  = _mode(0, spin)
            bath_mode = _mode(b_site, spin)
            # hop: impurity -> bath
            H += FermionOperator(((imp_mode, 1), (bath_mode, 0)),  V_b)
            # hop: bath -> impurity (Hermitian conjugate)
            H += FermionOperator(((bath_mode, 1), (imp_mode, 0)),  V_b)

    return H


#def hubbard_hamiltonian(t, U, nx, ny):

import openfermion as of
from openfermion import FermionOperator
import numpy as np


def tV_hamiltonian_2d(nx: int, ny: int, t: float, V: float, mu: float = 0.0,
                      periodic_x: bool = True, periodic_y: bool = True):
    """
    Build the spinless t-V Hamiltonian on an nx x ny 2D lattice.

    H = -t * sum_{<i,j>} (c†_i c_j + h.c.)
      + V * sum_{<i,j>} n_i n_j
      - mu * sum_i n_i

    Args:
        nx:         Number of sites in x-direction.
        ny:         Number of sites in y-direction.
        t:          Hopping amplitude.
        V:          Nearest-neighbor interaction strength.
        mu:         Chemical potential (default 0).
        periodic_x: Use periodic boundary conditions in x (default True).
        periodic_y: Use periodic boundary conditions in y (default True).

    Returns:
        hamiltonian: FermionOperator representing the t-V Hamiltonian.
        site_index:  Function mapping (ix, iy) -> linear site index.
    """
    n_sites = nx * ny

    def site_index(ix: int, iy: int) -> int:
        """Map 2D coordinates to a linear site index."""
        return iy * nx + ix

    hamiltonian = FermionOperator()

    # ------------------------------------------------------------------
    # Hopping and interaction terms along x-bonds
    # ------------------------------------------------------------------
    for iy in range(ny):
        for ix in range(nx):
            # Determine right neighbor
            ix_right = ix + 1
            if ix_right == nx:
                if not periodic_x:
                    continue           # open boundary: skip edge bond
                ix_right = 0           # wrap around

            i = site_index(ix, iy)
            j = site_index(ix_right, iy)

            # Hopping: -t (c†_i c_j + c†_j c_i)
            hamiltonian += FermionOperator(f'{i}^ {j}', -t)
            hamiltonian += FermionOperator(f'{j}^ {i}', -t)

            # Interaction: V n_i n_j
            hamiltonian += FermionOperator(f'{i}^ {i} {j}^ {j}', V)

    # ------------------------------------------------------------------
    # Hopping and interaction terms along y-bonds
    # ------------------------------------------------------------------
    for iy in range(ny):
        for ix in range(nx):
            # Determine upper neighbor
            iy_up = iy + 1
            if iy_up == ny:
                if not periodic_y:
                    continue           # open boundary: skip edge bond
                iy_up = 0              # wrap around

            i = site_index(ix, iy)
            j = site_index(ix, iy_up)

            # Hopping: -t (c†_i c_j + c†_j c_i)
            hamiltonian += FermionOperator(f'{i}^ {j}', -t)
            hamiltonian += FermionOperator(f'{j}^ {i}', -t)

            # Interaction: V n_i n_j
            hamiltonian += FermionOperator(f'{i}^ {i} {j}^ {j}', V)

    # ------------------------------------------------------------------
    # Chemical potential: -mu * sum_i n_i
    # ------------------------------------------------------------------
    if mu != 0.0:
        for i in range(n_sites):
            hamiltonian += FermionOperator(f'{i}^ {i}', -mu)

    return hamiltonian, site_index

from openfermion import FermionOperator


def tV_hamiltonian_1d(L, t=1.0, V=1.0, mu=0.0, periodic=False):
    """
    Construct 1D spinless t-V Hamiltonian.

    H = -t Σ (c†_i c_{i+1} + h.c.)
        + V Σ n_i n_{i+1}
        - μ Σ n_i
    """

    H = FermionOperator()

    # hopping terms
    for i in range(L - 1):
        H += -t * FermionOperator(f"{i}^ {i+1}")
        H += -t * FermionOperator(f"{i+1}^ {i}")

    # periodic boundary hopping
    if periodic and L > 2:
        H += -t * FermionOperator(f"{L-1}^ 0")
        H += -t * FermionOperator(f"0^ {L-1}")

    # interaction terms
    for i in range(L - 1):
        H += V * FermionOperator(f"{i}^ {i} {i+1}^ {i+1}")

    if periodic and L > 2:
        H += V * FermionOperator(f"{L-1}^ {L-1} 0^ 0")

    # chemical potential
    for i in range(L):
        H += -mu * FermionOperator(f"{i}^ {i}")

    return H

import openfermion as of

def site_index(x, y, Nx, spin):
    """
    Maps a site (x, y) and spin (0 for up, 1 for down)
    to a single fermionic mode index.

    Index = 2*(y*Nx + x) + spin
    where spin in {0, 1}.
    Total number of modes = 2 * Nx * Ny.
    """
    return 2 * (y * Nx + x) + spin


def hubbard_2d(Nx, Ny, t, U, mu=0.0, periodic=False):
    """
    Returns the FermionOperator for the 2D Hubbard model on
    an Nx x Ny lattice.

    Parameters
    ----------
    Nx, Ny   : int   — number of sites along x and y directions
    t        : float — hopping amplitude (enters as -t)
    U        : float — on-site Coulomb interaction strength
    mu       : float — chemical potential (enters as -mu * n)
    periodic : bool  — if True, impose periodic boundary conditions

    Returns
    -------
    H : FermionOperator
    """
    H = of.FermionOperator()

    # ----------------------------------------------------------------
    # 1. Hopping terms: -t * sum_{<i,j>} sum_sigma (c†_i,sigma c_j,sigma + h.c.)
    # ----------------------------------------------------------------
    # Nearest-neighbor pairs in x-direction: (x,y) -- (x+1, y)
    # Nearest-neighbor pairs in y-direction: (x,y) -- (x, y+1)

    def add_hopping(i, j):
        """Adds -t*(c†_i c_j + c†_j c_i) for each spin."""
        nonlocal H
        for spin in (0, 1):
            p = site_index(*i, Nx, spin)
            q = site_index(*j, Nx, spin)
            # c†_p c_q
            H += of.FermionOperator(((p, 1), (q, 0)), -t)
            # c†_q c_p  (Hermitian conjugate)
            H += of.FermionOperator(((q, 1), (p, 0)), -t)

    for y in range(Ny):
        for x in range(Nx):
            # x-direction neighbor
            x_right = (x + 1) % Nx if periodic else x + 1
            if periodic or x + 1 < Nx:
                add_hopping((x, y), (x_right, y))
            # y-direction neighbor
            y_up = (y + 1) % Ny if periodic else y + 1
            if periodic or y + 1 < Ny:
                add_hopping((x, y), (x, y_up))

    # ----------------------------------------------------------------
    # 2. On-site interaction: U * sum_i n_{i,up} * n_{i,down}
    #
    # n_{i,up} * n_{i,down} = c†_{i,up} c_{i,up} c†_{i,down} c_{i,down}
    # ----------------------------------------------------------------
    for y in range(Ny):
        for x in range(Nx):
            p_up   = site_index(x, y, Nx, 0)   # spin-up mode index
            p_down = site_index(x, y, Nx, 1)   # spin-down mode index
            H += of.FermionOperator(
                ((p_up, 1), (p_up, 0), (p_down, 1), (p_down, 0)),
                U
            )

    # ----------------------------------------------------------------
    # 3. Chemical potential: -mu * sum_i sum_sigma n_{i,sigma}
    # ----------------------------------------------------------------
    if mu != 0.0:
        for y in range(Ny):
            for x in range(Nx):
                for spin in (0, 1):
                    p = site_index(x, y, Nx, spin)
                    H += of.FermionOperator(((p, 1), (p, 0)), -mu)

    return H


# ---------------------------------------------------------------
# Example: 2x2 lattice, t=1, U=4, mu=0, open boundary conditions
# ---------------------------------------------------------------
Nx, Ny = 2, 2
t, U, mu = 1.0, 4.0, 0.0

H = hubbard_2d(Nx, Ny, t, U, mu, periodic=False)
print(H)

# Convert to sparse matrix (dimension = 4^(Nx*Ny) = 256 for 2x2)
sparse_H = of.get_sparse_operator(H)
print("Hilbert space dimension:", sparse_H.shape)
# ----------------------------------------------------------------------
# Quick sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from openfermion.utils import count_qubits

    N = 3
    H = build_siam_hamiltonian(N=N, U=4.0, t=1.0, V=1.0, eps_imp=0.0)

    n_modes = 2 * (N + 1)
    print(f"N = {N} bath sites")
    print(f"Total spin-orbitals (modes): {n_modes}")
    print(f"Number of terms in FermionOperator: {len(H.terms)}")
    print()
    print("All terms:")
    for term, coeff in sorted(H.terms.items(), key=lambda x: (len(x[0]), x[0])):
        print(f"  {coeff:+.6f}  *  {term}")
    print(H)
