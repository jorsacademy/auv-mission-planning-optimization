"""Risk-aware convoy escort allocation package."""

from .model import AllocationResult, Convoy, Escort, solve_allocation
from .uncertainty import (
    CVaRAllocationResult,
    ScenarioOutcome,
    ThreatScenario,
    UncertainAllocationResult,
    solve_cvar_allocation,
    solve_uncertain_allocation,
)

__all__ = [
    "AllocationResult",
    "CVaRAllocationResult",
    "Convoy",
    "Escort",
    "ScenarioOutcome",
    "ThreatScenario",
    "UncertainAllocationResult",
    "solve_allocation",
    "solve_cvar_allocation",
    "solve_uncertain_allocation",
]
