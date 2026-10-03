import numpy as np
import scipy as sci
import symengine as se
import fieldmath as fm
import dimod, itertools
from math import sqrt
from openfermion.linalg import get_sparse_operator
from openfermion.ops import QubitOperator
from openfermion.transforms import taper_off_qubits
from openfermion.utils import count_qubits

'''
Helper functions to construct Pauli operator Hamiltonian
'''


# Returns molecular geometry for a given molecule and bond length
def get_molGeometry(name, BL):
    geometries = {
        'H2': [['H', (0, 0, 0)], ['H', (BL, 0, 0)]],
        'H3': [['H', (0, 0, 0)], ['H', (BL, 0, 0)], ['H', (BL / 2, sqrt(3) / 2 * BL, 0)]],
        'LiH': [['Li', (0, 0, 0)], ['H', (BL, 0, 0)]],
        'CH4': [['C', (0, 0, 0)], ['H', (-BL / sqrt(3), BL / sqrt(3), BL / sqrt(3))],
                ['H', (BL / sqrt(3), -BL / sqrt(3), BL / sqrt(3))],
                ['H', (BL / sqrt(3), BL / sqrt(3), -BL / sqrt(3))],
                ['H', (-BL / sqrt(3), -BL / sqrt(3), -BL / sqrt(3))]],
        'H2O': [['O', (0, 0, 0)], ['H', (BL, 0, 0)], ['H', (np.cos(1.823478115) * BL, np.sin(1.823478115) * BL, 0)]],
        'OH': [['O', (0, 0, 0)], ['H', (BL, 0, 0)]],
        'HeH': [['He', (0, 0, 0)], ['H', (BL, 0, 0)]],
        'O2': [['O', (0, 0, 0)], ['O', (BL, 0, 0)]],
        'N2': [['N', (0, 0, 0)], ['N', (BL, 0, 0)]],
        'CO2': [['C', (0, 0, 0)], ['O', (-BL, 0, 0)], ['O', (BL, 0, 0)]],
        'HCl': [['H', (0, 0, 0)], ['Cl', (BL, 0, 0)]],
        'NH3': [['N', (0, 0, 0)], ['H', (BL, 0, 0)], ['H', (-BL / 2, np.sqrt(3) / 2 * BL, 0)],
                ['H', (-BL / 2, -np.sqrt(3) / 2 * BL, 0)]],
        'SF': [['S', (0, 0, 0)], ['F', (BL, 0, 0)]],
        'Si': [['Si', (0, 0, 0)]],
        'H2S': [['S', (0, 0, 0)], ['H', (BL, 0, 0)], ['H', (-BL / 2, np.sqrt(3) / 2 * BL, 0)]],
        'NO2': [['N', (0, 0, 0)], ['O', (BL, 0, 0)], ['O', (-BL * np.cos(1.823478115), BL * np.sin(1.823478115), 0)]],
        'Cl2': [['Cl', (0, 0, 0)], ['Cl', (BL, 0, 0)]],
        'C': [['C', (0, 0, 0)]],
        'CO': [['C', (0, 0, 0)], ['O', (BL, 0, 0)]],
        'UO': [['U', (0, 0, 0)], ['O', (BL, 0, 0)]],

    }
    return geometries[name]


# Returns indices of doubly occupied and active orbitals
def get_active_space(molecule, n_active_electrons, n_active_orbitals):
    n_occupied_orbitals = (molecule.n_electrons - n_active_electrons) // 2
    occupied_indices = list(range(n_occupied_orbitals))
    active_indices = list(range(n_occupied_orbitals, n_occupied_orbitals + n_active_orbitals))

    return occupied_indices, active_indices

'''
Helper functions to reduce qubit count
'''


# Utilize symmetries to split Hamiltonian into sectors
def taper_qubits(qubit_H):
    n = count_qubits(qubit_H)
    H_dict = qubit_H.terms
    terms = list(H_dict.keys())
    num_terms = len(terms)

    # create bit string representation of each term
    field = fm.PrimeField(2)
    E = fm.Matrix(num_terms, 2 * n, field)
    for i in range(num_terms):

        term = terms[i]
        spot = 0
        for j in range(n):
            try:
                if term[spot][0] == j:
                    char = term[spot][1]
                    spot += 1
                else:
                    char = 'I'
            except IndexError:
                char = 'I'

            if char == 'I':
                E.set(i, j, 0)
                E.set(i, j + n, 0)
            if char == 'X':
                E.set(i, j, 0)
                E.set(i, j + n, 1)
            if char == 'Y':
                E.set(i, j, 1)
                E.set(i, j + n, 1)
            if char == 'Z':
                E.set(i, j, 1)
                E.set(i, j + n, 0)

    E.reduced_row_echelon_form()
    E_reduced = np.empty((num_terms, 2 * n), dtype=int)
    for row in range(num_terms):
        for col in range(2 * n):
            E_reduced[row][col] = E.get(row, col)
    del E

    while (E_reduced[-1] == np.zeros(2 * n)).all():
        E_reduced = np.delete(E_reduced, len(E_reduced) - 1, axis=0)

    # determine nullspace of parity matrix
    pivots, first_entries = [], []
    E_reduced = E_reduced.transpose()
    for col in range(len(E_reduced)):
        try:
            first_entry = list(E_reduced[col]).index(1)
            isPivot = True
            for col2 in range(col):
                if E_reduced[col2][first_entry] == 1:
                    isPivot = False
            if isPivot:
                pivots += [col]
                first_entries += [first_entry]
        except ValueError:
            pass
    nonpivots = list(set(range(len(E_reduced))) - set(pivots))

    nullspace = []
    for col in nonpivots:
        col_vector = list(E_reduced[col])

        null_vector = [0] * 2 * n
        for i in range(2 * n):
            if col == i:
                null_vector[i] = 1
            elif i in pivots:
                first_entry = first_entries[pivots.index(i)]
                if col_vector[first_entry] == 1:
                    null_vector[i] = 1
        nullspace += [null_vector]
    del E_reduced

    # create symmetry generators
    generators = []
    for i in range(len(nullspace)):
        null_vector = nullspace[i]
        tau = ''
        for j in range(n):
            x = null_vector[j]
            z = null_vector[j + n]

            if x == 0 and z == 0:
                tau += 'I'
            elif x == 1 and z == 0:
                tau += 'X'
            elif x == 1 and z == 1:
                tau += 'Y'
            else:
                tau += 'Z'
        generators += [tau]

    # convert generators into QubitOperators
    for i in range(len(generators)):
        tau = generators[i]

        tau_str = ''
        for j in range(n):
            if tau[j] != 'I':
                tau_str += tau[j] + str(j) + ' '

        generators[i] = QubitOperator(tau_str)

    # use generators to create different sectors of Hamiltonian
    sectors = []
    perms = list(itertools.product([1, -1], repeat=len(generators)))
    for perm in perms:
        signed_generators = [perm[i] * generators[i] for i in range(len(generators))]

        sector = taper_off_qubits(qubit_H, signed_generators)
        sector.compress()
        sectors += [sector]

    return sectors


# Returns the sector with the smallest eigenvalue via brute force
def sector_with_ground(sectors, return_eigenvalue=True):
    min_eigenvalues = []

    for sector in sectors:
        sparse_H = get_sparse_operator(sector).todense()

        if count_qubits(sector) <= 2:
            min_eigenvalue = min(sci.linalg.eigvals(sparse_H))
        else:
            min_eigenvalue = sci.sparse.linalg.eigsh(sparse_H, k=1, which='SA', return_eigenvectors=False)
        min_eigenvalues += [float(min_eigenvalue.real)]

    index = min_eigenvalues.index(min(min_eigenvalues))

    if return_eigenvalue:
        return sectors[index], min_eigenvalues[index]
    else:
        return sectors[index]


'''
Helper functions for XBK method
'''


# Convert dictionary from OpenFermion form to dimod form
def convert_dict(dictionary):
    new_dict = {}
    for key in dictionary:
        var_list = []
        for var in key:
            var_list += ['s' + str(var[0])]
        var_list = tuple(var_list)

        new_dict[var_list] = dictionary[key]
    return new_dict


# Convert a dimod dictionary into a function using symengine
def dict_to_func(dictionary):
    expr = 0
    for key in dictionary:
        term = dictionary[key]
        for var in key:
            term *= se.Symbol(var)
        expr += term

    if type(expr) == float:
        f = expr
    else:
        var_list = list(expr.free_symbols)
        var_list.sort(key=sort_disc_func)
        f = se.lambdify(var_list, (expr,))
    return f


'''
Helper functions for QCC method
'''


# Sort function for discrete variables
def sort_disc_func(variable):
    return int(str(variable)[1:])


# Sort function for continuous variables
def sort_cont_func(variable):
    return int(str(variable)[3:])


# Sorts mixed list of discrete and continuous variables into seperate sorted lists
def sort_mixed_vars(var_list):
    Z_vars, Q_vars, W_vars, F_vars, G_vars = [], [], [], [], []
    phi_vars, the_vars, tau_vars = [], [], []

    for i in range(len(var_list)):
        variable = var_list[i]

        if str(variable)[0] == 'Z':
            Z_vars += [variable]
        elif str(variable)[0] == 'Q':
            Q_vars += [variable]
        elif str(variable)[0] == 'W':
            W_vars += [variable]
        elif str(variable)[0] == 'F':
            F_vars += [variable]
        elif str(variable)[0] == 'G':
            G_vars += [variable]

        elif str(variable)[:3] == 'phi':
            phi_vars += [variable]
        elif str(variable)[:3] == 'the':
            the_vars += [variable]
        elif str(variable)[:3] == 'tau':
            tau_vars += [variable]

    Z_vars.sort(key=sort_disc_func)
    Q_vars.sort(key=sort_disc_func)
    W_vars.sort(key=sort_disc_func)
    F_vars.sort(key=sort_disc_func)
    G_vars.sort(key=sort_disc_func)
    phi_vars.sort(key=sort_cont_func)
    the_vars.sort(key=sort_cont_func)
    tau_vars.sort(key=sort_cont_func)

    return (Z_vars + Q_vars + W_vars + F_vars + G_vars, phi_vars + the_vars + tau_vars)


# Determine the bounds of a continuous variable list
def get_bounds(cont_vars, angle_folds=0, amplitude_folds=0):
    bounds = []
    for var in cont_vars:
        if str(var)[:3] == 'phi':
            if angle_folds < 2:
                bounds += [(0, 2 * np.pi)]
            elif angle_folds == 2:
                bounds += [(0, np.pi)]
            else:
                bounds += [(0, np.pi / 2)]

        elif str(var)[:3] == 'the':
            if angle_folds == 0:
                bounds += [(0, np.pi)]
            else:
                bounds += [(0, np.pi / 2)]

        elif str(var)[:3] == 'tau':
            bounds += [(0, 2 * np.pi / (2 ** amplitude_folds))]
    return np.array(bounds)


# Converts a symengine expression into a dictionary
def expr_to_dict(expr):
    expr2 = se.lib.symengine_wrapper.Add(expr)
    terms = se.Add.make_args(expr2)

    dictionary = {}
    for term in terms:
        variables = tuple(term.free_symbols)
        try:
            coeff = float(se.Mul.make_args(term)[0])
        except RuntimeError:
            coeff = 1

        dictionary[variables] = coeff
    return dictionary


# Converts a QubitOperator into a symengine expression
def qubit_op_to_expr(qubit_op, angle_folds=0):
    qubit_op.compress()
    dict_op = qubit_op.terms

    expr = 0
    for key in dict_op:
        term = dict_op[key]

        for var in key:
            num, char = var

            if char == 'X':
                term *= se.cos(se.Symbol('phi' + str(num))) * se.sin(se.Symbol('the' + str(num)))
                if angle_folds == 3:
                    term *= se.Symbol('W' + str(num))
            if char == 'Y':
                term *= se.sin(se.Symbol('phi' + str(num))) * se.sin(se.Symbol('the' + str(num)))
                if angle_folds > 1:
                    term *= se.Symbol('Q' + str(num))
            if char == 'Z':
                term *= se.cos(se.Symbol('the' + str(num)))
                if angle_folds > 0:
                    term *= se.Symbol('Z' + str(num))
        expr += term
    return expr


import pandas as pd
import dimod, math, itertools
from openfermion.ops import QubitOperator, FermionOperator
from openfermion.utils import count_qubits
from openfermion import bravyi_kitaev
#from helper_functions import *
from dwave.samplers import SimulatedAnnealingSampler


# Applies XBK transformation to an OpenFermion QubitOperator
def XBK_transform(op, r, p):
    n = count_qubits(op)
    op_terms = op.terms
    new_op = QubitOperator()

    # transform operator term by term
    for key in op_terms:
        coeff = op_terms[key]
        term = QubitOperator()

        # cycle through each of the r ancillarly qubit copies
        for j in range(r):
            for k in range(r):
                sign = 1 if (j < p) == (k < p) else -1
                sub_term = QubitOperator('', 1)

                # cycle through each of the n original qubits
                spot = 0
                for i in range(n):
                    try:
                        if key[spot][0] == i:
                            char = key[spot][1]
                            spot += 1
                        else:
                            char = 'I'
                    except IndexError:
                        char = 'I'

                    # use variable type to apply correct mapping
                    if char == 'X':
                        if j == k:
                            sub_term = QubitOperator('', 0)
                            break
                        else:
                            sub_term *= QubitOperator('', 1 / 2) - QubitOperator(
                                'Z' + str(i + n * j) + ' Z' + str(i + n * k), 1 / 2)
                    elif char == 'Y':
                        if j == k:
                            sub_term = QubitOperator('', 0)
                            break
                        else:
                            sub_term *= QubitOperator('Z' + str(i + n * k), 1j / 2) - QubitOperator(
                                'Z' + str(i + n * j), 1j / 2)
                    elif char == 'Z':
                        if j == k:
                            sub_term *= QubitOperator('Z' + str(i + n * j), 1)
                        else:
                            sub_term *= QubitOperator('Z' + str(i + n * j), 1 / 2) + QubitOperator('Z' + str(i + n * k),
                                                                                                   1 / 2)
                    else:
                        if j == k:
                            continue
                        else:
                            sub_term *= QubitOperator('', 1 / 2) + QubitOperator(
                                'Z' + str(i + n * j) + ' Z' + str(i + n * k), 1 / 2)

                term += sign * sub_term
        new_op += coeff * term

    new_op.compress()
    return new_op


# Construct C term required for XBK method
def construct_C(n, r, p):
    C = QubitOperator('', 0)
    perms = list(itertools.product([1, -1], repeat=n))
    k = 0
    for perm in perms:
        term = QubitOperator('', 0)
        print(k + 1)
        k += 1
        for j in range(r):
            product = QubitOperator('', 1)

            for i in range(n):
                product *= QubitOperator('', 1 / 2) + QubitOperator('Z' + str(i + n * j), perm[i] / 2)

            sign = -1 if j < p else 1
            term += sign * product
        C += term ** 2
    return C


# Find minimum energy and ground state using XBK method
def XBK(qubit_Hs, qubit_Cs, r, sampler, starting_lam=0, num_samples=1000, strength=1e3, verbose=False):
    n = count_qubits(qubit_Hs[0])
    min_energies, ground_states = [], []

    for p in range(int(math.ceil(r / 2 + 1))):
        qubit_H, qubit_C = qubit_Hs[p], qubit_Cs[p]

        # create C function to evalute sum(b^2)
        C_dict = convert_dict(qubit_C.terms)
        C_func = dict_to_func(C_dict)

        # calculate minimum energy for particular p value
        lam = starting_lam
        eigenvalue = min_energy = -1
        ground_state = []
        cycles = 0
        while min_energy < 0 and cycles < 10:
            # subtract lambda C from H
            H_prime = qubit_H - lam * qubit_C

            # construct qubo from reduced Hamiltonian
            bqm = dimod.higherorder.utils.make_quadratic(convert_dict(H_prime.terms), strength, dimod.SPIN)
            print(bqm)
            qubo, constant = bqm.to_qubo()

            if qubo == {}:
                break

            # run sampler
            response = sampler.sample_qubo(qubo, num_reads=num_samples)
            solutions = pd.DataFrame(response.data())

            # get mininum energy solution
            index = int(solutions[['energy']].idxmin())
            min_energy = round(solutions['energy'][index], 14) + constant
            full_solution = solutions['sample'][index]

            solution = []
            for key in ['s' + str(i) for i in range(n)]:
                try:
                    solution += [2 * full_solution[key] - 1]
                except KeyError:
                    solution += [0]
                    print('KeyError: ' + str(key))

            # calculate sum(b^2)
            try:
                sumBsq = int(C_func(*solution))
            except TypeError:
                sumBsq = int(C_func)

            if sumBsq == 0:  # stop the loop in zero case
                cycles += 1
                break

            # calculate the eigenvalue of H
            eigenvalue = lam + min_energy / sumBsq

            # set lam equal to eigenvalue for next cycle
            if min_energy < 0:
                lam = eigenvalue
                ground_state = [(val + 1) // 2 for val in solution]
            cycles += 1

        min_energies += [round(lam, 14)]
        ground_states += [ground_state]

        if verbose:
            print('P:', p, 'E:', round(lam, 5))

    index = min_energies.index(min(min_energies))
    min_energy = min_energies[index]
    ground_state = ground_states[index]

    if verbose:
        print('Energy:', round(min_energy, 5))

    return min_energy, ground_state


from openfermion import count_qubits
from dwave.samplers import SimulatedAnnealingSampler

H_qubit = QubitOperator('Y1 Z2 Z4')
n = count_qubits(H_qubit)  # n=5 (qubits 0-4)
r = 3

# precompute H and C for each p value
qubit_Hs = []
qubit_Cs = []
for p in range(int(math.ceil(r / 2 + 1))):
    qubit_Hs.append(XBK_transform(H_qubit, r=r, p=p))
    qubit_Cs.append(construct_C(n, r=r, p=p))

# instantiate sampler
sampler = SimulatedAnnealingSampler()

# run
energy, state = XBK(
    qubit_Hs,
    qubit_Cs,
    r=r,
    sampler=sampler,
    starting_lam=100
)
print("Ground state energy:", energy)
print("Ground state:", state)