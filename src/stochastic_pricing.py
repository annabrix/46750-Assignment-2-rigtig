"""Question 3: dynamic pricing when the members' preferences are uncertain.

Scenarios come from src/scenario_generation.py (the rules of preference_uncertainty_params.json):
each member's reference profile is rho_t ** theta, and its daily minimum follows the profile.

    draw_scenarios      joint scenarios of the community (seeded, aligned merging)
    scenario_members    the members as one scenario sees them, ready for the models of Q1 and Q2
    reference_costs     every member's cost in R, per scenario (the IR benchmark)
    build_sdp           the stochastic model SDP^x: shared prices, one lower level per member and scenario
    solve_sdp           solve it; per-scenario results and KKT residual checks
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
from .dynamic_pricing import add_kkt, best_response


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

def reference_costs(com, scenarios: list[dict]) -> dict:
    """Cost of every member alone without contract (model R) in every scenario: {(scenario id, consumer id): DKK}."""
    return {(s["id"], m.consumer_id): CommunityModel([m]).build(pooled=False).solve().objective
            for s in scenarios for m in scenario_members(com, s)}


def build_sdp(com, scenarios: list[dict], R_s: dict | None = None, *, requirements: str = "bb",
              how: str = "expectation", min_relax: bool = False, discount: float | None = None, cap=None,
              x_lb=0.0, x_ub=np.inf, end_soc: str = "initial", verbose: bool = False) -> dict:
    """Stochastic dynamic pricing SDP: one price vector x for all scenarios, one copy of every member's
    lower level (its KKT conditions) and of the connection point per scenario.

    Switches of the parametrized model:
      scenarios     the scenario set (list of joint scenarios from draw_scenarios), with probabilities pi_s
      requirements  "cap" (contract only), "bb" (+ budget balance), "bb_ir" (+ individual rationality against R
                    in the same scenario, costs R_s from reference_costs), "bb_ir_relax" (IR with slacks >= 0)
      how           "expectation": the requirements hold in expectation over the scenarios (base model SDP^E);
                    "every": they hold in every scenario
      min_relax     objective = total IR relaxation (only with bb_ir_relax); otherwise the expected welfare cost
    The objective is the expected community cost sum_s pi_s F_s (risk-neutral).
    Prices: x_lb <= x_t <= x_ub (as in build_dp).
    """
    if requirements not in ("cap", "bb", "bb_ir", "bb_ir_relax"):
        raise ValueError(f"Unknown requirements {requirements!r}")
    if how not in ("expectation", "every"):
        raise ValueError("how must be 'expectation' or 'every'")
    if requirements in ("bb_ir", "bb_ir_relax") and R_s is None:
        raise ValueError("Individual rationality needs the R costs per scenario (reference_costs)")

    ct, d0 = com.contract, com.members[0]
    beta = ct.tariff_discount_ratio if discount is None else float(discount)
    alpha = ct.penalty_DKK_per_kWh
    p = np.asarray(d0.energy_price, dtype=float)
    T = range(len(p))
    cap = np.broadcast_to(np.asarray(ct.capacity_limit_kW if cap is None else cap, dtype=float), (len(T),))
    a = p + (1 - beta) * d0.import_tariff
    b = p - (1 - beta) * d0.export_tariff
    lo = np.broadcast_to(np.asarray(x_lb, dtype=float), (len(T),))
    hi = a + alpha if x_ub is None else np.broadcast_to(np.asarray(x_ub, dtype=float), (len(T),))

    m = gp.Model("SDP")
    m.Params.OutputFlag = int(verbose)
    x = m.addVars(T, lb=[float(v) for v in lo], ub=[float(v) for v in hi], name="x")   # first stage

    S = [s["id"] for s in scenarios]
    pi = {s["id"]: s["probability"] for s in scenarios}
    blocks, kkt, D, phi, pay, imp, exp, exceed, grid, F = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for sc in scenarios:                                                                 # second stage
        s = sc["id"]
        for d in scenario_members(com, sc):
            k = (s, d.consumer_id)
            blocks[k] = add_member_block(m, d, f"{d.consumer_id},s{s}", end_soc)
            D[k] = blocks[k]["disutility"] + blocks[k]["pv_cost"]
        imp[s] = m.addVars(T, name=f"P_imp[s{s}]")
        exp[s] = m.addVars(T, name=f"P_exp[s{s}]")
        exceed[s] = m.addVars(T, name=f"E[s{s}]")
        m.addConstrs((imp[s][t] - exp[s][t] == gp.quicksum(blocks[k]["q"][t] for k in blocks if k[0] == s)
                      for t in T), name=f"community_balance[s{s}]")
        m.addConstrs((imp[s][t] - exceed[s][t] <= cap[t] for t in T), name=f"DSO_cap[s{s}]")
        grid[s] = gp.quicksum(a[t] * imp[s][t] - b[t] * exp[s][t] + alpha * exceed[s][t] for t in T)
        F[s] = grid[s] + gp.quicksum(D[k] for k in blocks if k[0] == s)

    for k, blk in blocks.items():
        kkt[k] = add_kkt(m, blk, x, f"{k[1]},s{k[0]}")
        phi[k] = kkt[k]["phi"]                      # member cost (strong duality)
        pay[k] = phi[k] - D[k]                      # member payment x.q
    ids = sorted({k[1] for k in blocks})
    expected_cost = gp.quicksum(pi[s] * F[s] for s in S)

    if requirements != "cap":
        surplus = {s: gp.quicksum(pay[k] for k in blocks if k[0] == s) - grid[s] for s in S}
        if how == "expectation":
            m.addConstr(gp.quicksum(pi[s] * surplus[s] for s in S) == 0, name="budget_balance")
        else:
            m.addConstrs((surplus[s] == 0 for s in S), name="budget_balance")

    relax = {}
    if requirements in ("bb_ir", "bb_ir_relax"):
        for i in ids:
            if requirements == "bb_ir_relax":
                relax[i] = m.addVar(lb=0.0, name=f"IR_relax[{i}]")
            if how == "expectation":
                m.addConstr(gp.quicksum(pi[s] * (phi[(s, i)] - R_s[(s, i)]) for s in S) <= relax.get(i, 0.0),
                            name=f"IR[{i}]")
            else:
                m.addConstrs((phi[(s, i)] - R_s[(s, i)] <= relax.get(i, 0.0) for s in S), name=f"IR[{i}]")

    if min_relax and requirements != "bb_ir_relax":
        raise ValueError("min_relax requires requirements='bb_ir_relax'")
    m.setObjective(gp.quicksum(relax.values()) if min_relax else expected_cost, GRB.MINIMIZE)
    m.update()
    return {"model": m, "x": x, "scenarios": S, "pi": pi, "ids": ids, "blocks": blocks, "kkt": kkt, "D": D,
            "phi": phi, "pay": pay, "imp": imp, "exp": exp, "exceed": exceed, "grid": grid, "F": F, "cap": cap,
            "relax": relax, "expected_cost": expected_cost, "requirements": requirements, "how": how}


def solve_sdp(bundle, *, time_limit: float = 600, mip_gap: float = 1e-4, x_start=None, R_s: dict | None = None,
              verbose: bool = False) -> dict:
    """Solve an SDP model from build_sdp; return the prices, per-scenario results and the KKT residual checks.

    x_start: optional 24 prices (e.g. the DP prices) given to Gurobi as a partial MIP start.
    R_s: optional R costs per scenario, to report the members worse off than R in every scenario.
    """
    m = bundle["model"]
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)
    if x_start is not None:
        for t, v in enumerate(np.asarray(x_start, dtype=float)):
            bundle["x"][t].Start = v
    m.optimize()

    status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT",
              GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED"}.get(m.Status, str(m.Status))
    out = {"status": status, "runtime_s": m.Runtime, "has_solution": m.SolCount > 0, "n_vars": m.NumVars,
           "n_constrs": m.NumConstrs, "n_sos": m.NumSOS, "n_scenarios": len(bundle["scenarios"]),
           "mip_gap": float(m.MIPGap) if m.SolCount else np.nan,
           "best_bound": float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else np.nan}
    if not out["has_solution"]:
        return out

    T = range(len(bundle["x"]))
    x = np.array([bundle["x"][t].X for t in T])
    rows, max_pay, max_comp, max_stat = [], 0.0, 0.0, 0.0
    for s in bundle["scenarios"]:
        E = np.array([bundle["exceed"][s][t].X for t in T])
        payments, worse, violation = 0.0, 0, 0.0
        for i in bundle["ids"]:
            k = (s, i)
            q = np.array([bundle["blocks"][k]["q"][t].X for t in T])
            pay = bundle["pay"][k].getValue()
            payments += pay
            max_pay = max(max_pay, abs(pay - x @ q))
            max_comp = max([max_comp] + [abs(mu.X * sl.X) for mu, sl in bundle["kkt"][k]["pairs"]])
            max_stat = max([max_stat] + [abs(c.Slack) for c in bundle["kkt"][k]["stations"]])
            if R_s is not None:
                gap = bundle["phi"][k].getValue() - R_s[k]
                if gap > 1e-5:
                    worse, violation = worse + 1, violation + gap
        grid = bundle["grid"][s].getValue()
        rows.append({"scenario": s, "probability": bundle["pi"][s], "welfare_cost_DKK": bundle["F"][s].getValue(),
                     "grid_bill_DKK": grid, "surplus_DKK": payments - grid, "exceedance_kWh": E.sum(),
                     "hours_above_cap": int((E > 1e-6).sum()),
                     "n_worse_off_than_R": worse if R_s is not None else np.nan,
                     "total_IR_violation_DKK": violation if R_s is not None else np.nan})
    out.update({"x": x, "per_scenario": pd.DataFrame(rows), "expected_cost_DKK": bundle["expected_cost"].getValue(),
                "solver_objective": m.ObjVal, "max_payment_gap": max_pay, "max_stationarity": max_stat,
                "max_complementarity": max_comp})
    return out