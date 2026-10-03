import openfermion as op
import numpy as np

def tV_1D(L, t, V, periodic=False):
    H = op.FermionOperator()
    if periodic:
        for i in range(L):


def tV_2D(nx, ny, t, V, periodicx=False, periodicy=False):
    H = op.FermionOperator()
    return

def hubbard_1D(L, t, U, periodic=False):
    H = op.FermionOperator()
    return

def hubbard_2d(nx, ny, t, U, periodicx=False, periodicy=False):
    H = op.FermionOperator()
    return

def SIAM(Nbath, t, V, U):
    H = op.FermionOperator()
    for spin in range(2):
        for k in range(Nbath):

    return

def PAM(Nbath, t, V, U, x):
    return

def J1J2_1D(J1, J2):
    return

def J1J2_2D(J1x, J1y, J2x, J2y):
    return

def kitaev_chain():
    return