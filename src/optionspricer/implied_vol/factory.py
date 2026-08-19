"""Registry-based factory for IV solvers, same rationale as
`pricing/factory.py`: new solvers register themselves instead of this file
growing an if/elif chain."""

from __future__ import annotations  # postpones type-hint evaluation, same rationale as market.py

from optionspricer.implied_vol.base import IVSolver  # the common type every registry value is a subclass of
from optionspricer.implied_vol.brent import BrentSolver  # imported here (not lazily) so registration below can run at import time
from optionspricer.implied_vol.jaeckel import JaeckelSolver
from optionspricer.implied_vol.newton import NewtonSolver

_REGISTRY: dict[str, type[IVSolver]] = {}  # name -> class, not name -> instance; instantiated fresh on every create_iv_solver call


def register_solver(name: str, solver_cls: type[IVSolver]) -> None:  # called once per solver, at the bottom of this file
    _REGISTRY[name] = solver_cls  # mutates the module-level dict; no return value needed


def create_iv_solver(name: str, **kwargs) -> IVSolver:  # **kwargs forwards constructor args, e.g. tol=1e-6
    try:
        solver_cls = _REGISTRY[name]  # KeyError here means the name was never registered
    except KeyError:
        raise ValueError(f"unknown IV solver {name!r}; available: {sorted(_REGISTRY)}") from None  # `from None` suppresses the KeyError traceback, since it's not useful context for the caller
    return solver_cls(**kwargs)  # instantiate the class, passing through whatever kwargs the caller supplied


def available_solvers() -> list[str]:  # what experiments/run_iv_solver_benchmark.py loops over instead of hardcoding a list
    return sorted(_REGISTRY)  # sorted() on a dict iterates its keys; alphabetical order makes output reproducible


register_solver("newton", NewtonSolver)  # these three calls run once, when this module is first imported
register_solver("brent", BrentSolver)
register_solver("jaeckel", JaeckelSolver)
