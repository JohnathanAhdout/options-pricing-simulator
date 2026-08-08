"""Registry-based factory for IV solvers, same rationale as
`pricing/factory.py`: new solvers register themselves instead of this file
growing an if/elif chain."""

from __future__ import annotations

from optionspricer.implied_vol.base import IVSolver
from optionspricer.implied_vol.brent import BrentSolver
from optionspricer.implied_vol.jaeckel import JaeckelSolver
from optionspricer.implied_vol.newton import NewtonSolver

_REGISTRY: dict[str, type[IVSolver]] = {}


def register_solver(name: str, solver_cls: type[IVSolver]) -> None:
    _REGISTRY[name] = solver_cls


def create_iv_solver(name: str, **kwargs) -> IVSolver:
    try:
        solver_cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown IV solver {name!r}; available: {sorted(_REGISTRY)}") from None
    return solver_cls(**kwargs)


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)


register_solver("newton", NewtonSolver)
register_solver("brent", BrentSolver)
register_solver("jaeckel", JaeckelSolver)
