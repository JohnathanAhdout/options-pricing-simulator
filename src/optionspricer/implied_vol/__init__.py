from optionspricer.implied_vol.base import IVSolver  # the interface and its three implementations, re-exported so callers
from optionspricer.implied_vol.brent import BrentSolver  # can `from optionspricer.implied_vol import BrentSolver` instead
from optionspricer.implied_vol.factory import available_solvers, create_iv_solver, register_solver  # of reaching into each submodule directly
from optionspricer.implied_vol.jaeckel import JaeckelSolver
from optionspricer.implied_vol.newton import NewtonSolver

__all__ = [  # controls `from optionspricer.implied_vol import *` and documents the package's public surface
    "IVSolver",
    "NewtonSolver",
    "BrentSolver",
    "JaeckelSolver",
    "create_iv_solver",
    "register_solver",
    "available_solvers",
]
