"""Question 3: dynamic pricing when the members' preferences are uncertain.

Scenarios come from src/scenario_generation.py (the rules of preference_uncertainty_params.json):
each member's reference profile is rho_t ** theta, and its daily minimum follows the profile.

    draw_scenarios      joint scenarios of the community (seeded, aligned merging)
    scenario_members    the members as one scenario sees them, ready for the models of Q1 and Q2
"""
from __future__ import annotations

import copy

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from . import scenario_generation as sg
from .community_model import CommunityModel, add_member_block
from .dynamic_pricing import best_response


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

def respond(com, members, x, *, discount: float | None = None, cap=None, tie_break: str = "optimistic",
            tol: float = 1e-6) -> dict:
    """The members' responses to fixed posted prices x (24 values, DKK/kWh), and the community's flows.

    Each member first solves its own LP at the prices x (best_response), which gives its optimal cost v_i.
    tie_break="optimistic": among all schedules that are optimal for every member (cost_i <= v_i + tol), pick the
    ones that minimize the community's welfare cost, as the manager anticipates in DP. With x fixed this is one LP.
    tie_break="solver": keep the schedule the LP solver happened to return for each member (an arbitrary choice
    where a member is indifferent).
    Returns the net consumption q per member, the member costs and payments, and the flows at the meter.
    """
    ct, d0 = com.contract, com.members[0]
    beta = ct.tariff_discount_ratio if discount is None else float(discount)
    alpha = ct.penalty_DKK_per_kWh
    p = np.asarray(d0.energy_price, dtype=float)
    T = range(len(p))
    cap = np.broadcast_to(np.asarray(ct.capacity_limit_kW if cap is None else cap, dtype=float), (len(T),))
    a = p + (1 - beta) * d0.import_tariff
    b = p - (1 - beta) * d0.export_tariff
    x = np.asarray(x, dtype=float)

    br = {m.consumer_id: best_response(m, x) for m in members}
    if tie_break == "solver":
        q = {i: r["q"] for i, r in br.items()}
        cost = {i: r["cost"] for i, r in br.items()}
    elif tie_break == "optimistic":
        mdl = gp.Model("optimistic_response")
        mdl.Params.OutputFlag = 0
        blocks = {m.consumer_id: add_member_block(mdl, m, m.consumer_id) for m in members}
        for i, blk in blocks.items():
            own = gp.quicksum(x[t] * blk["q"][t] for t in T) + blk["disutility"] + blk["pv_cost"]
            mdl.addConstr(own <= br[i]["cost"] + tol * max(1.0, abs(br[i]["cost"])), name=f"optimal_face[{i}]")
        imp, exp, exc = mdl.addVars(T, name="P_imp"), mdl.addVars(T, name="P_exp"), mdl.addVars(T, name="E")
        mdl.addConstrs((imp[t] - exp[t] == gp.quicksum(blk["q"][t] for blk in blocks.values()) for t in T))
        mdl.addConstrs((imp[t] - exc[t] <= cap[t] for t in T))
        grid = gp.quicksum(a[t] * imp[t] - b[t] * exp[t] + alpha * exc[t] for t in T)
        mdl.setObjective(grid + gp.quicksum(blk["disutility"] + blk["pv_cost"] for blk in blocks.values()),
                         GRB.MINIMIZE)
        mdl.optimize()
        if mdl.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimistic response ended with status {mdl.Status}")
        q = {i: np.array([blk["q"][t].X for t in T]) for i, blk in blocks.items()}
        cost = {i: float(x @ q[i]) + (blk["disutility"] + blk["pv_cost"]).getValue() for i, blk in blocks.items()}
    else:
        raise ValueError("tie_break must be 'optimistic' or 'solver'")

    Q = sum(q.values())
    P_imp, P_exp = np.maximum(Q, 0.0), np.maximum(-Q, 0.0)
    over = np.maximum(P_imp - cap, 0.0)
    grid = float(a @ P_imp - b @ P_exp + alpha * over.sum())
    pay = {i: float(x @ q[i]) for i in q}
    return {"q": q, "cost": cost, "payment": pay, "P_imp": P_imp, "P_exp": P_exp, "exceedance": over,
            "grid_bill": grid, "penalty": alpha * over.sum(),
            "welfare_cost": grid + sum(cost[i] - pay[i] for i in q)}


def evaluate_prices(com, x, scenarios: list[dict], *, tie_break: str = "optimistic", discount: float | None = None,
                    cap=None) -> tuple:
    """Evaluate posted prices x (24 values, DKK/kWh) on joint scenarios of the members' preferences.

    In every scenario, the members respond to the fixed prices x on their scenario data (respond, with the chosen
    tie_break), and each member's cost alone without contract (model R) is computed on the same data.
    Returns two DataFrames:
      per scenario: welfare cost, grid bill, penalty (DKK); exceedance (kWh), hours above the cap, peak import (kW);
                    manager's surplus = payments - grid bill (DKK); members worse off than R and their total loss (DKK)
      per scenario and member: theta, cost (payment + disutility + PV cost), payment, cost in R, gain vs R (DKK)
    """
    rows, member_rows = [], []
    for s in scenarios:
        members = scenario_members(com, s)
        r = respond(com, members, x, discount=discount, cap=cap, tie_break=tie_break)
        gains = []
        for m in members:
            i = m.consumer_id
            cost_R = CommunityModel([m]).build(pooled=False).solve().objective
            gains.append(cost_R - r["cost"][i])
            member_rows.append({"scenario": s["id"], "consumer_id": i, "theta": s["theta"]["profile"][i],
                                "cost_DKK": r["cost"][i], "payment_DKK": r["payment"][i], "cost_R_DKK": cost_R,
                                "gain_vs_R_DKK": gains[-1]})
        gains = np.array(gains)
        rows.append({"scenario": s["id"], "probability": s["probability"],
                     "welfare_cost_DKK": r["welfare_cost"], "grid_bill_DKK": r["grid_bill"],
                     "penalty_DKK": r["penalty"], "exceedance_kWh": r["exceedance"].sum(),
                     "hours_above_cap": int((r["exceedance"] > 1e-6).sum()), "peak_import_kW": r["P_imp"].max(),
                     "surplus_DKK": sum(r["payment"].values()) - r["grid_bill"],
                     "n_worse_off_than_R": int((gains < -1e-5).sum()),
                     "total_IR_violation_DKK": float(-gains[gains < -1e-5].sum())})
    return pd.DataFrame(rows), pd.DataFrame(member_rows)