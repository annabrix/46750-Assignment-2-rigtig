"""Question 3: dynamic pricing when the members' preferences are uncertain.

Scenarios come from src/scenario_generation.py (the rules of preference_uncertainty_params.json):
each member's reference profile is rho_t ** theta, and its daily minimum follows the profile.

    draw_scenarios      joint scenarios of the community (seeded, aligned merging)
    scenario_members    the members as one scenario sees them, ready for the models of Q1 and Q2
    StochasticDP        SDP^x: one price vector, one KKT system per member and scenario (switches: objective, budget, IR)
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

class StochasticDP:
    """SDP^x: dynamic pricing with uncertain member preferences, as one parametrized model.

    One price vector x_t (first stage, the same in every scenario); in every scenario s the members respond
    optimally to x with their scenario data (second stage: one KKT system per member and scenario, as in DP),
    and the community trades at the meter. Optimistic solution concept, as in DP.

        sdp = StochasticDP(com, scenarios)           # scenarios from draw_scenarios
        sdp.build(objective="expected", budget="expectation", ir="none")
        res = sdp.solve(time_limit=300)

    Switches of build():
        objective  "expected": expected community cost (the base model SDP^b)
                   "cvar": (1 - beta_r) * expected cost + beta_r * CVaR_alpha of the cost (SDP^CVaR);
                           alpha = fraction of worst scenarios (a multiple of 1/N), beta_r = weight in [0, 1]
        budget     "none" | "expectation" | "scenario": payments = grid bill, in expectation or in every scenario
                   "chance": budget balance in expectation, and no deficit with probability >= 1 - eps on the
                             scenarios (SDP^CC; one binary per scenario, big-M = big_m DKK)
        ir         "none" | "expectation" | "scenario": member cost <= its cost in R on the same scenario data
        x_lb, x_ub price bounds (DKK/kWh), default x >= 0 without upper bound, as in DP
        reg        weight of a small regularization term: reg * expected |surplus|. The cost does not depend on the
                   price level, so many price vectors are optimal; the term picks the one with the most stable
                   budget among them (default 1e-3 DKK per DKK, negligible against the cost)
    """

    def __init__(self, com, scenarios: list[dict], *, R: dict | None = None, name: str = "SDP"):
        self.com, self.name = com, name
        self.scenarios = scenarios
        self.members = {s["id"]: scenario_members(com, s) for s in scenarios}
        self.pi = {s["id"]: float(s["probability"]) for s in scenarios}
        self.R = R                      # {(scenario, consumer_id): cost in R (DKK)}, computed when needed

    def reference_costs(self) -> dict:
        """Cost of every member alone without contract (model R), on every scenario's data."""
        if self.R is None:
            self.R = {(s, m.consumer_id): CommunityModel([m]).build(pooled=False).solve().objective
                      for s, ms in self.members.items() for m in ms}
        return self.R

    def build(self, *, objective: str = "expected", budget: str = "expectation", ir: str = "none",
              alpha: float = 0.2, beta_r: float = 0.5, eps: float = 0.2, big_m: float = 50.0, reg: float = 1e-3,
              deficit_tol: float = 0.0,
              x_lb=0.0, x_ub=np.inf, discount: float | None = None, cap=None,
              verbose: bool = False) -> "StochasticDP":
        if objective not in ("expected", "cvar"):
            raise ValueError("objective must be 'expected' or 'cvar'")
        if budget not in ("none", "expectation", "scenario", "chance"):
            raise ValueError("budget must be 'none', 'expectation', 'scenario' or 'chance'")
        if ir not in ("none", "expectation", "scenario"):
            raise ValueError("ir must be 'none', 'expectation' or 'scenario'")
        ct, d0 = self.com.contract, self.com.members[0]
        beta = ct.tariff_discount_ratio if discount is None else float(discount)
        pen = ct.penalty_DKK_per_kWh               # alpha^DSO (the name alpha is the CVaR tail fraction)
        p = np.asarray(d0.energy_price, dtype=float)
        T = range(len(p))
        cap = np.broadcast_to(np.asarray(ct.capacity_limit_kW if cap is None else cap, dtype=float), (len(T),))
        a = p + (1 - beta) * d0.import_tariff
        b = p - (1 - beta) * d0.export_tariff
        lo = np.broadcast_to(np.asarray(x_lb, dtype=float), (len(T),))
        hi = a + pen if x_ub is None else np.broadcast_to(np.asarray(x_ub, dtype=float), (len(T),))

        m = gp.Model(self.name)
        m.Params.OutputFlag = int(verbose)
        x = m.addVars(T, lb=[float(v) for v in lo], ub=[float(v) for v in hi], name="x")

        S = list(self.members)
        blocks, kkt, phi, D, pay, grid, cost, imp, exp, exc = ({} for _ in range(10))
        for s in S:
            blocks[s] = {d.consumer_id: add_member_block(m, d, f"{d.consumer_id}|{s}") for d in self.members[s]}
            D[s] = {i: blk["disutility"] + blk["pv_cost"] for i, blk in blocks[s].items()}
            imp[s] = m.addVars(T, name=f"P_imp[{s}]")
            exp[s] = m.addVars(T, name=f"P_exp[{s}]")
            exc[s] = m.addVars(T, name=f"E[{s}]")
            m.addConstrs((imp[s][t] - exp[s][t] == gp.quicksum(blk["q"][t] for blk in blocks[s].values())
                          for t in T), name=f"community_balance[{s}]")
            m.addConstrs((imp[s][t] - exc[s][t] <= cap[t] for t in T), name=f"DSO_cap[{s}]")
            grid[s] = gp.quicksum(a[t] * imp[s][t] - b[t] * exp[s][t] + pen * exc[s][t] for t in T)
            cost[s] = grid[s] + gp.quicksum(D[s].values())
            kkt[s] = {i: add_kkt(m, blk, x, f"{i}|{s}") for i, blk in blocks[s].items()}
            phi[s] = {i: kkt[s][i]["phi"] for i in blocks[s]}          # member cost (strong duality)
            pay[s] = {i: phi[s][i] - D[s][i] for i in blocks[s]}       # member payment x.q, linear

        # budget balance: payments collected = grid bill, in expectation or in every scenario
        surplus = {s: gp.quicksum(pay[s].values()) - grid[s] for s in S}
        if budget == "expectation":
            m.addConstr(gp.quicksum(self.pi[s] * surplus[s] for s in S) == 0, name="budget_balance_expected")
        elif budget == "scenario":
            m.addConstrs((surplus[s] == 0 for s in S), name="budget_balance")
        v_def = {}
        if budget == "chance":
            # expected balance, and a deficit allowed only in scenarios with v_s = 1, at most eps of them
            m.addConstr(gp.quicksum(self.pi[s] * surplus[s] for s in S) == 0, name="budget_balance_expected")
            v_def = {s: m.addVar(vtype=GRB.BINARY, name=f"deficit_allowed[{s}]") for s in S}
            m.addConstrs((surplus[s] >= -deficit_tol - big_m * v_def[s] for s in S), name="no_deficit")
            m.addConstr(gp.quicksum(self.pi[s] * v_def[s] for s in S) <= eps + 1e-9, name="deficit_probability")

        # individual rationality against R on the same scenario data
        if ir != "none":
            R = self.reference_costs()
            ids = [d.consumer_id for d in self.com.members]
            if ir == "expectation":
                m.addConstrs((gp.quicksum(self.pi[s] * (phi[s][i] - R[(s, i)]) for s in S) <= 0 for i in ids),
                             name="IR_expected")
            else:
                m.addConstrs((phi[s][i] <= R[(s, i)] for s in S for i in ids), name="IR")

        expected_cost = gp.quicksum(self.pi[s] * cost[s] for s in S)
        # regularization: |surplus_s| <= dev_s, so that among equally cheap prices the most stable budget is chosen
        dev = {s: m.addVar(lb=0.0, name=f"surplus_dev[{s}]") for s in S}
        m.addConstrs((dev[s] >= surplus[s] for s in S), name="dev_pos")
        m.addConstrs((dev[s] >= -surplus[s] for s in S), name="dev_neg")
        regularization = reg * gp.quicksum(self.pi[s] * dev[s] for s in S)
        zeta, eta, cvar = None, {}, None
        if objective == "cvar":
            # linear CVaR (Rockafellar-Uryasev): zeta = VaR threshold, eta_s = cost of scenario s above it
            zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")
            eta = {s: m.addVar(lb=0.0, name=f"eta[{s}]") for s in S}
            m.addConstrs((eta[s] >= cost[s] - zeta for s in S), name="cvar")
            cvar = zeta + (1.0 / alpha) * gp.quicksum(self.pi[s] * eta[s] for s in S)
            m.setObjective((1 - beta_r) * expected_cost + beta_r * cvar + regularization, GRB.MINIMIZE)
        else:
            m.setObjective(expected_cost + regularization, GRB.MINIMIZE)
        m.update()
        self.m, self.x, self.T, self.cap = m, x, T, cap
        self.v = {"blocks": blocks, "kkt": kkt, "phi": phi, "D": D, "pay": pay, "grid": grid, "cost": cost,
                  "imp": imp, "exp": exp, "exc": exc, "surplus": surplus, "expected_cost": expected_cost,
                  "cvar": cvar, "v_def": v_def}
        self.settings = {"objective": objective, "budget": budget, "ir": ir, "reg": reg}
        if objective == "cvar":
            self.settings.update(alpha=alpha, beta_r=beta_r)
        if budget == "chance":
            self.settings.update(eps=eps, big_m=big_m, deficit_tol=deficit_tol)
        return self

    def solve(self, *, time_limit: float = 300, mip_gap: float = 1e-4, x_start=None,
              verbose: bool = False) -> dict:
        """Solve and return the prices, the per-scenario results, the KKT residuals and the model size.

        x_start: optional 24 prices (e.g. of DP or SDP^b). The model is first solved with the prices fixed to
        x_start (the members' responses only, which is fast), and that solution is the MIP start of the full solve.
        """
        m, T, v = self.m, self.T, self.v
        m.Params.OutputFlag = int(verbose)
        m.Params.MIPGap = float(mip_gap)
        start_time = 0.0
        if x_start is not None:
            bounds = [(self.x[t].LB, self.x[t].UB) for t in T]
            for t in T:
                self.x[t].LB = self.x[t].UB = float(x_start[t])
            m.Params.TimeLimit = min(120.0, float(time_limit))
            m.optimize()
            start_time = m.Runtime
            values = [(var, var.X) for var in m.getVars()] if m.SolCount else []
            for t in T:
                self.x[t].LB, self.x[t].UB = bounds[t]
            m.reset(0)
            for var, val in values:
                var.Start = val
            if not values:                   # fixed prices infeasible (e.g. budget): give the prices as a partial start
                for t in T:
                    self.x[t].Start = float(x_start[t])
        m.Params.TimeLimit = float(time_limit)
        m.optimize()
        status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT",
                  GRB.INF_OR_UNBD: "INF_OR_UNBD"}.get(m.Status, str(m.Status))
        out = {"status": status, **self.settings, "N": len(self.members), "runtime_s": start_time + m.Runtime,
               "n_vars": m.NumVars, "n_constrs": m.NumConstrs, "n_sos": m.NumSOS,
               "has_solution": m.SolCount > 0, "mip_gap": float(m.MIPGap) if m.SolCount else np.nan,
               "best_bound": float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else np.nan}
        if not out["has_solution"]:
            return out

        x = np.array([self.x[t].X for t in T])
        R = self.reference_costs()
        rows, max_res = [], 0.0
        for s in self.members:
            exc = np.array([v["exc"][s][t].X for t in T])
            gains = np.array([R[(s, i)] - v["phi"][s][i].getValue() for i in v["phi"][s]])
            for i, k in v["kkt"][s].items():
                q = np.array([v["blocks"][s][i]["q"][t].X for t in T])
                max_res = max([max_res, abs(v["pay"][s][i].getValue() - x @ q)]
                              + [abs(mu.X * sl.X) for mu, sl in k["pairs"]] + [abs(c.Slack) for c in k["stations"]])
            rows.append({"scenario": s, "probability": self.pi[s], "cost_DKK": v["cost"][s].getValue(),
                         "grid_bill_DKK": v["grid"][s].getValue(), "exceedance_kWh": exc.sum(),
                         "penalty_DKK": self.com.contract.penalty_DKK_per_kWh * exc.sum(),
                         "surplus_DKK": v["surplus"][s].getValue(),
                         "deficit_allowed": int(round(v["v_def"][s].X)) if v["v_def"] else np.nan,
                         "n_worse_off_than_R": int((gains < -1e-5).sum()),
                         "total_IR_violation_DKK": float(-gains[gains < -1e-5].sum())})
        scen = pd.DataFrame(rows)
        out.update({"x": x, "objective_DKK": m.ObjVal, "expected_cost_DKK": v["expected_cost"].getValue(),
                    "cvar_DKK": v["cvar"].getValue() if v["cvar"] is not None else np.nan,
                    "max_KKT_residual": max_res, "scenarios": scen})
        if v["v_def"] and (-scen.surplus_DKK).max() > 0.9 * self.settings["big_m"]:
            raise RuntimeError("A deficit is close to big_m: increase big_m")
        return out