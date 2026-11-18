"""Question 2: dynamic pricing (model DP) for the energy community.

The members' constraints are the ones of Question 1 (add_member_block in community_model.py),
so they are written in one place only.

    best_response   one member's LP at fixed internal prices x_t (the lower level of DP)
    add_kkt         one member's KKT conditions (SOS1 complementarity), for the single-level reformulation
    build_dp        the single-level MILP of DP, with the variants cap / bb / bb_ir / bb_ir_relax
    solve_dp        solve it and check the KKT residuals; solution_values gives a warm start
"""
from __future__ import annotations

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .community_model import add_member_block


def best_response(d, x, end_soc: str = "initial") -> dict:
    """Lower level of DP: member d schedules its appliances against the internal prices x.

    Minimizes  sum_t x_t q_t + disutility + PV cost  subject to the member's constraints of Question 1.
    x: 24 internal prices (DKK/kWh). Returns the member's cost (DKK), its net consumption q and its
    load (kWh/h), and the Gurobi model and member block (used later for the indifference check).
    """
    x = np.asarray(x, dtype=float)
    T = range(len(x))
    m = gp.Model(f"BR_{d.consumer_id}")
    m.Params.OutputFlag = 0
    b = add_member_block(m, d, d.consumer_id, end_soc)
    m.setObjective(gp.quicksum(x[t] * b["q"][t] for t in T) + b["disutility"] + b["pv_cost"], GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Best response of {d.consumer_id} ended with status {m.Status}")
    return {
        "cost": m.ObjVal,
        "q": np.array([b["q"][t].X for t in T]),
        "load": np.array([b["var"]["load"][t].X for t in T]),
        "model": m,
        "block": b,
    }
    
def _complementarity(m, multiplier, slack_expr, name):
    """multiplier >= 0, slack >= 0 and multiplier * slack = 0, as an SOS1 pair (no big-M needed)."""
    slack = m.addVar(lb=0.0, name=f"slack[{name}]")
    m.addConstr(slack == slack_expr, name=f"define_slack[{name}]")
    m.addSOS(GRB.SOS_TYPE1, [multiplier, slack])
    return slack


def add_kkt(m, block, x, tag: str) -> dict:
    """Replace one member's LP (its block from add_member_block) by its KKT conditions in model m.

    x: the 24 internal prices, either numbers or upper-level Gurobi variables. Adds, for every
    constraint and variable bound of the member: its multiplier (dual feasibility), the
    complementarity pair as SOS1, and one stationarity equation per primal variable.
    Returns phi, the member's dual objective: by strong duality it equals the member's optimal
    cost x.q + disutility + PV cost, and it is linear in the multipliers.
    """
    m.update()
    primal = [v for family in block["var"].values() for v in family.values()]

    rows = []
    for name, con in block["con"].items():
        if isinstance(con, gp.tupledict):
            rows += [(f"{name},{k}", c) for k, c in con.items()]
        else:
            rows.append((name, con))

    # gradient of the member's objective: disutility + PV cost + x_t on q_t
    gradient = {v.index: gp.LinExpr() for v in primal}
    cost = block["disutility"] + block["pv_cost"]
    for k in range(cost.size()):
        gradient[cost.getVar(k).index] += cost.getCoeff(k)
    for t in range(len(block["q"])):
        gradient[block["q"][t].index] += x[t]

    phi, pairs, stations = gp.LinExpr(), [], []

    # multipliers of the explicit constraints
    for name, con in rows:
        lhs, rhs, sense = m.getRow(con), float(con.RHS), con.Sense
        if sense == "=":
            mu = m.addVar(lb=-GRB.INFINITY, name=f"pi[{tag},{name}]")
            sign = 1
            phi += -rhs * mu
        elif sense == "<":
            mu = m.addVar(lb=0.0, name=f"mu_le[{tag},{name}]")
            sign = 1
            phi += -rhs * mu
            pairs.append((mu, _complementarity(m, mu, rhs - lhs, f"{tag},{name}")))
        else:  # ">"
            mu = m.addVar(lb=0.0, name=f"mu_ge[{tag},{name}]")
            sign = -1
            phi += rhs * mu
            pairs.append((mu, _complementarity(m, mu, lhs - rhs, f"{tag},{name}")))
        for k in range(lhs.size()):
            gradient[lhs.getVar(k).index] += sign * lhs.getCoeff(k) * mu

    # multipliers of the variable bounds, and one stationarity equation per variable
    for v in primal:
        if v.LB > -GRB.INFINITY / 2:
            mu = m.addVar(lb=0.0, name=f"mu_lb[{tag},{v.VarName}]")
            gradient[v.index] += -mu
            phi += float(v.LB) * mu
            pairs.append((mu, _complementarity(m, mu, v - float(v.LB), f"{tag},lb,{v.VarName}")))
        if v.UB < GRB.INFINITY / 2:
            mu = m.addVar(lb=0.0, name=f"mu_ub[{tag},{v.VarName}]")
            gradient[v.index] += mu
            phi += -float(v.UB) * mu
            pairs.append((mu, _complementarity(m, mu, float(v.UB) - v, f"{tag},ub,{v.VarName}")))
        stations.append(m.addConstr(gradient[v.index] == 0, name=f"stationarity[{tag},{v.VarName}]"))

    return {"phi": phi, "pairs": pairs, "stations": stations}

def build_dp(com, variant: str = "cap", R: dict | None = None, *, discount: float | None = None, cap=None,
             x_lb=0.0, x_ub=np.inf, min_relax: bool = False, end_soc: str = "initial",
             verbose: bool = False) -> dict:
    """Single-level MILP of DP: the manager's problem with every member's LP replaced by its KKT conditions.

    Optimistic solution concept: where a member is indifferent, the manager picks the response it prefers.
    Variants (switch ``variant``):
        cap          the contract only (cap, penalty, discount at the connection point)
        bb           + budget balance: members' payments = the community's bill at the connection point
        bb_ir        + individual rationality: every member's cost <= its cost in R (dict R, DKK)
        bb_ir_relax  as bb_ir, but member i may violate IR by IR_relax[i] >= 0 (DKK)
    min_relax: objective = total IR relaxation instead of welfare (only with bb_ir_relax).
    """
    if variant not in ("cap", "bb", "bb_ir", "bb_ir_relax"):
        raise ValueError(f"Unknown variant {variant!r}")
    if variant in ("bb_ir", "bb_ir_relax") and R is None:
        raise ValueError("Individual rationality needs the cost of every member in R")
    if min_relax and variant != "bb_ir_relax":
        raise ValueError("min_relax requires variant='bb_ir_relax'")

    ct, d0 = com.contract, com.members[0]
    beta = ct.tariff_discount_ratio if discount is None else float(discount)
    alpha = ct.penalty_DKK_per_kWh
    p = np.asarray(d0.energy_price, dtype=float)
    T = range(len(p))
    cap = np.broadcast_to(np.asarray(ct.capacity_limit_kW if cap is None else cap, dtype=float), (len(T),))
    a = p + (1 - beta) * d0.import_tariff        # price of a kWh imported at the meter
    b = p - (1 - beta) * d0.export_tariff        # price of a kWh exported at the meter
    lo = np.broadcast_to(np.asarray(x_lb, dtype=float), (len(T),))
    hi = a + alpha if x_ub is None else np.broadcast_to(np.asarray(x_ub, dtype=float), (len(T),))

    m = gp.Model(f"DP_{variant}")
    m.Params.OutputFlag = int(verbose)

    # upper level: the posted prices, the same for every member
    x = m.addVars(T, lb=[float(v) for v in lo], ub=[float(v) for v in hi], name="x")

    # lower levels: the members' constraints of Question 1, then their KKT conditions
    blocks = {d.consumer_id: add_member_block(m, d, d.consumer_id, end_soc) for d in com.members}
    D = {i: blk["disutility"] + blk["pv_cost"] for i, blk in blocks.items()}

    # connection point, as in model C
    imp = m.addVars(T, name="P_imp")
    exp = m.addVars(T, name="P_exp")
    exceed = m.addVars(T, name="E")
    m.addConstrs((imp[t] - exp[t] == gp.quicksum(blk["q"][t] for blk in blocks.values()) for t in T),
                 name="community_balance")
    m.addConstrs((imp[t] - exceed[t] <= cap[t] for t in T), name="DSO_cap")
    grid = gp.quicksum(a[t] * imp[t] - b[t] * exp[t] + alpha * exceed[t] for t in T)
    welfare = grid + gp.quicksum(D.values())

    kkt = {i: add_kkt(m, blocks[i], x, i) for i in blocks}

    # strong duality: member cost = phi (linear), so its payment x.q = phi - D is linear too
    phi = {i: kkt[i]["phi"] for i in blocks}
    pay = {i: phi[i] - D[i] for i in blocks}

    if variant != "cap":
        m.addConstr(gp.quicksum(pay.values()) == grid, name="budget_balance")

    relax = {}
    if variant in ("bb_ir", "bb_ir_relax"):
        for i in blocks:
            if variant == "bb_ir_relax":
                relax[i] = m.addVar(lb=0.0, name=f"IR_relax[{i}]")
            m.addConstr(phi[i] <= float(R[i]) + relax.get(i, 0.0), name=f"IR[{i}]")

    m.setObjective(gp.quicksum(relax.values()) if min_relax else welfare, GRB.MINIMIZE)
    m.update()
    return {"model": m, "variant": variant, "x": x, "blocks": blocks, "kkt": kkt, "imp": imp, "exp": exp,
            "exceed": exceed, "cap": cap, "grid": grid, "D": D, "phi": phi, "pay": pay, "relax": relax,
            "welfare": welfare}


def solution_values(bundle) -> dict:
    """Values of all variables of a solved DP model, by name: a warm start for another DP model."""
    return {v.VarName: v.X for v in bundle["model"].getVars()}


def solve_dp(bundle, *, time_limit: float = 300, mip_gap: float = 1e-4, start: dict | None = None,
             verbose: bool = False) -> dict:
    """Solve a DP model from build_dp and return its results and the KKT residual checks.

    start: optional {variable name: value}, e.g. from solution_values(), given to Gurobi as a MIP start.
    """
    m = bundle["model"]
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)
    if start:
        for v in m.getVars():
            if v.VarName in start:
                v.Start = start[v.VarName]
    m.optimize()

    status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT",
              GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED"}.get(m.Status, str(m.Status))
    out = {"status": status, "variant": bundle["variant"], "runtime_s": m.Runtime,
           "has_solution": m.SolCount > 0, "n_vars": m.NumVars, "n_constrs": m.NumConstrs, "n_sos": m.NumSOS,
           "mip_gap": float(m.MIPGap) if m.SolCount else np.nan,
           "best_bound": float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else np.nan}
    if not out["has_solution"]:
        return out

    T = range(len(bundle["x"]))
    x = np.array([bundle["x"][t].X for t in T])
    hourly = pd.DataFrame({"hour": list(T), "x": x,
                           "P_imp": [bundle["imp"][t].X for t in T], "P_exp": [bundle["exp"][t].X for t in T],
                           "E": [bundle["exceed"][t].X for t in T], "cap": bundle["cap"]})

    rows, max_pay, max_comp, max_stat = [], 0.0, 0.0, 0.0
    for i, blk in bundle["blocks"].items():
        q = np.array([blk["q"][t].X for t in T])
        payment = bundle["pay"][i].getValue()
        max_pay = max(max_pay, abs(payment - x @ q))
        max_comp = max([max_comp] + [abs(mu.X * s.X) for mu, s in bundle["kkt"][i]["pairs"]])
        max_stat = max([max_stat] + [abs(c.Slack) for c in bundle["kkt"][i]["stations"]])
        rows.append({"consumer_id": i, "cost_DKK": bundle["phi"][i].getValue(), "payment_DKK": payment,
                     "physical_cost_DKK": bundle["D"][i].getValue(),
                     "IR_relax_DKK": bundle["relax"][i].X if i in bundle["relax"] else 0.0,
                     "net_kWh": q.sum(), **{f"q_{t}": q[t] for t in T}})
    members = pd.DataFrame(rows)
    grid = bundle["grid"].getValue()
    out.update({"objective_DKK": bundle["welfare"].getValue(), "solver_objective": m.ObjVal,
                "grid_cost_DKK": grid, "budget_surplus_DKK": members.payment_DKK.sum() - grid,
                "cap_exceedance_kWh": hourly.E.sum(), "peak_import_kW": hourly.P_imp.max(),
                "total_IR_relax_DKK": members.IR_relax_DKK.sum(), "members": members, "hourly": hourly,
                "max_payment_gap": max_pay, "max_stationarity": max_stat, "max_complementarity": max_comp})
    return out