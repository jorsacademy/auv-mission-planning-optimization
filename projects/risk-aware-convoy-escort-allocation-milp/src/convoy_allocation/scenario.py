from __future__ import annotations

from .model import Convoy, Escort
from .uncertainty import ThreatScenario


def build_demo_scenario() -> tuple[dict[str, Convoy], dict[str, Escort], int]:
    """Return a deterministic synthetic scenario for demonstration and testing."""
    convoys = {
        "Convoy-A": Convoy(ships=44, threat=8.0, baseline_survival=0.42, max_escorts=4),
        "Convoy-B": Convoy(ships=36, threat=6.5, baseline_survival=0.48, max_escorts=4),
        "Convoy-C": Convoy(ships=31, threat=9.0, baseline_survival=0.36, max_escorts=4),
        "Convoy-D": Convoy(ships=39, threat=5.5, baseline_survival=0.53, max_escorts=4),
        "Convoy-E": Convoy(ships=27, threat=10.0, baseline_survival=0.33, max_escorts=4),
    }

    escorts = {
        "Escort-01": Escort(protection=0.78),
        "Escort-02": Escort(protection=0.72),
        "Escort-03": Escort(protection=0.68),
        "Escort-04": Escort(protection=0.64),
        "Escort-05": Escort(protection=0.60),
        "Escort-06": Escort(protection=0.56),
        "Escort-07": Escort(protection=0.52),
        "Escort-08": Escort(protection=0.48),
        "Escort-09": Escort(protection=0.44),
        "Escort-10": Escort(protection=0.40),
    }

    max_available_escorts = 8
    return convoys, escorts, max_available_escorts


def build_uncertain_demo_scenarios() -> dict[str, ThreatScenario]:
    """Return a finite synthetic threat distribution for stochastic experiments."""
    return {
        "Low": ThreatScenario(
            probability=0.25,
            threat_multipliers={
                "Convoy-A": 0.82,
                "Convoy-B": 0.88,
                "Convoy-C": 0.86,
                "Convoy-D": 0.90,
                "Convoy-E": 0.84,
            },
        ),
        "Nominal": ThreatScenario(
            probability=0.50,
            threat_multipliers={
                "Convoy-A": 1.00,
                "Convoy-B": 1.00,
                "Convoy-C": 1.00,
                "Convoy-D": 1.00,
                "Convoy-E": 1.00,
            },
        ),
        "High": ThreatScenario(
            probability=0.25,
            threat_multipliers={
                "Convoy-A": 1.28,
                "Convoy-B": 1.18,
                "Convoy-C": 1.34,
                "Convoy-D": 1.22,
                "Convoy-E": 1.30,
            },
        ),
    }
