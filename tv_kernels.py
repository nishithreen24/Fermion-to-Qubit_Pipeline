#!/usr/bin/env python3
"""
tv_kernels.py -- ctypes loader for the C QMC kernels in tv_kernels.c.

Compiles the shared library on first import (or when the .c file is newer than
the .so) and exposes two functions with NumPy-array signatures:

    sse_sweep(...)  -> (n_final, nout)
    pimc_sweep(...) -> Eout

numba was the original plan but llvmlite has no wheel for Python 3.14, and the
directed-loop updates are inherently sequential (pointer-chasing through the
leg-link structure), which neither numba's nopython loops nor NumPy
vectorisation handle well.  A small C kernel behind ctypes needs no third-party
package, compiles with the stock cc/gcc on both macOS and the cluster, and is
the natural fit for this access pattern.

If compilation fails, HAVE_C is False and callers fall back to the validated
pure-Python implementation in tv_levy_sqa.py.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "tv_kernels.c")
_LIB = os.path.join(_HERE, "libtvkernels.so")

HAVE_C = False
_lib = None
BUILD_ERROR = None


def _needs_build():
    if not os.path.exists(_LIB):
        return True
    return os.path.getmtime(_SRC) > os.path.getmtime(_LIB)


def _build():
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-O3", "-shared", "-fPIC", "-o", _LIB, _SRC, "-lm"]
    # -march=native helps but is not portable across cluster node types
    if os.environ.get("TVK_NATIVE"):
        cmd.insert(2, "-march=native")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{proc.stderr.strip()}")


I64 = ctypes.c_int64
DBL = ctypes.c_double
U64 = ctypes.c_uint64
P64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
PDB = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")

try:
    if _needs_build():
        _build()
    _lib = ctypes.CDLL(_LIB)

    _lib.sse_sweep.restype = I64
    _lib.sse_sweep.argtypes = [
        I64, P64, I64,          # L, bonds, Nb
        DBL, DBL, DBL, DBL,     # beta, Wpar, Wad, Woff
        I64, P64, P64, I64,     # Mcut, opstr, spins, n_in
        I64, I64, PDB, U64,     # warmup, measures, nout, seed
    ]

    _lib.pimc_sweep.restype = None
    _lib.pimc_sweep.argtypes = [
        I64, P64, P64,          # nv, legs, link
        DBL, DBL, DBL,          # Wpar, Wad, Wao
        DBL, DBL, DBL, DBL,     # g_par, g_ad, g_ao, M
        I64, I64, PDB, U64,     # warmup, measures, Eout, seed
    ]
    HAVE_C = True
except Exception as exc:                      # noqa: BLE001 - report and fall back
    BUILD_ERROR = str(exc)
    HAVE_C = False


def sse_sweep(L, bonds, beta, Wpar, Wad, Woff, Mcut, opstr, spins, n_in,
              warmup, measures, seed):
    """Run warmup+measures SSE steps in C.  Returns (n_final, nout array).

    bonds : (Nb,2) int64;  opstr : (Mcut,) int64;  spins : (L,) int64
    opstr and spins are updated IN PLACE so state carries across calls.
    """
    if not HAVE_C:
        raise RuntimeError(f"C kernel unavailable: {BUILD_ERROR}")
    bonds = np.ascontiguousarray(bonds, dtype=np.int64).ravel()
    Nb = bonds.size // 2
    nout = np.empty(measures, dtype=np.float64)
    n_final = _lib.sse_sweep(L, bonds, Nb, beta, Wpar, Wad, Woff,
                             Mcut, opstr, spins, n_in,
                             warmup, measures, nout, seed)
    if n_final < 0:
        raise MemoryError("sse_sweep: C allocation failed")
    return int(n_final), nout


def pimc_sweep(legs, link, Wpar, Wad, Wao, g_par, g_ad, g_ao, M,
               warmup, measures, seed):
    """Run warmup+measures PIMC loop steps in C.  Returns Eout array.

    legs is updated IN PLACE.  nv is inferred as len(legs)//4.
    """
    if not HAVE_C:
        raise RuntimeError(f"C kernel unavailable: {BUILD_ERROR}")
    nv = legs.size // 4
    Eout = np.empty(measures, dtype=np.float64)
    _lib.pimc_sweep(nv, legs, link, Wpar, Wad, Wao,
                    g_par, g_ad, g_ao, float(M),
                    warmup, measures, Eout, seed)
    return Eout


if __name__ == "__main__":
    print(f"HAVE_C = {HAVE_C}")
    if HAVE_C:
        print(f"loaded {_LIB}")
    else:
        print(f"build error:\n{BUILD_ERROR}", file=sys.stderr)
