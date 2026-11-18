"""Question 3: dynamic pricing when the members' preferences are uncertain.

Scenarios come from src/scenario_generation.py (the rules of preference_uncertainty_params.json):
each member's reference profile is rho_t ** theta, and its daily minimum follows the profile.

    draw_scenarios      joint scenarios of the community (seeded, aligned merging)
    scenario_members    the members as one scenario sees them, ready for the models of Q1 and Q2
"""
from __future__ import annotations

import copy

import numpy as np

from . import scenario_generation as sg


def draw_scenarios(com, n: int, seed: int, quantities=("profile",), spec: dict | None = None) -> list[dict]:
    """n equiprobable joint scenarios (aligned merging) for the loaded community.

    quantities: the uncertain quantities, "profile" (base study) and/or "cL". spec: optional tuned
    hyper-parameters (sg.rules_to_spec + sg.set_rule); default the values of the uncertainty file.
    Each scenario: {"id", "probability", "index", "theta", "members"} (see sg.merge_scenarios).
    """
    rules = sg.load_json("preference_uncertainty_params.json")
    return sg.draw_preference_scenarios(com.raw["appliance_params"], com.raw["params"], rules,
                                        n=n, seed=seed, quantities=quantities, spec=spec)


def scenario_members(com, scenario: dict) -> list:
    """The members of the community as one joint scenario sees them.

    Returns copies of com.members in which the load parameters of the scenario replace the base ones:
    reference profile L_max * rho_t ** theta (kWh/h), hourly bounds max(0, ref - Delta) .. min(L_max, ref + Delta),
    daily minimum (= the scenario's reference energy, kWh) and disutility slope (DKK/kWh).
    PV, battery, prices and tariffs are unchanged.
    """
    out = []
    for d in com.members:
        load = next(a for a in scenario["members"][d.consumer_id].values() if "load_id" in a)
        L_max = float(load["max_load_KWh_per_hour"])
        ref = L_max * np.asarray(load["reference_load_capacity_factor"], dtype=float)
        m = copy.copy(d)                       # shallow copy: the arrays below are replaced, never modified
        m.reference_load = ref
        m.load_lower_kWh = np.maximum(0.0, ref - d.max_hourly_deviation_kWh)
        m.load_upper_kWh = np.minimum(L_max, ref + d.max_hourly_deviation_kWh)
        m.min_daily_energy_kWh = float(load["min_total_energy_per_day_kWh"])
        m.linear_disutility = float(load["linear_disutility_DKK_per_kWh"])
        out.append(m)
    return out