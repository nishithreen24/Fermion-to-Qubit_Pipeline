from openfermion import FermionOperator, bravyi_kitaev, symmetry_conserving_bravyi_kitaev
import numpy as np
from prev_code import *
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


N_bath = 2
print(bravyi_kitaev(build_siam_hamiltonian(N_bath, 1.0, 1.0, 1.0, 0.0)))
print()
print(symmetry_conserving_bravyi_kitaev(build_siam_hamiltonian(N = N_bath, U = 1.0, t = 1.0, V = 1.0, eps_imp = 0.0), active_orbitals=2*(N_bath+1),
    active_fermions=N_bath+1))
print()