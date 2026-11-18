
"""Question 2: Dynamic pricing for an energy community.

Central benchmark, individual best responses, and optimistic
bilevel reformulation using KKT and SOS1 constraints.

Uses ordinary functions, not a new model class.
"""

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


# ============================================================
# STEP 1: PHYSICAL CONSUMER MODEL
# ============================================================

def add_member_block(m, d, end_soc="initial"):
    """Add one member's physical variables and constraints."""

    i = d.consumer_id
    T = range(len(d.energy_price))

    ref = np.asarray(d.reference_load, float)
    lo = np.asarray(d.load_lower_kWh, float)
    hi = np.asarray(d.load_upper_kWh, float)
    available = np.asarray(d.pv_available, float)

    battery = (
        d.battery_capacity_kWh is not None
        and d.battery_capacity_kWh > 0
    )

    v = {
        "load": m.addVars(T, name=f"load[{i}]"),
        "up": m.addVars(T, name=f"dev_up[{i}]"),
        "down": m.addVars(T, name=f"dev_down[{i}]"),
        "pv": m.addVars(T, name=f"pv[{i}]"),
        "q": m.addVars(
            T, lb=-GRB.INFINITY, name=f"q[{i}]"
        ),
    }

    if battery:
        v["charge"] = m.addVars(
            T,
            ub=d.battery_max_charge_kW,
            name=f"charge[{i}]"
        )
        v["discharge"] = m.addVars(
            T,
            ub=d.battery_max_discharge_kW,
            name=f"discharge[{i}]"
        )
        v["soc"] = m.addVars(
            T,
            ub=d.battery_capacity_kWh,
            name=f"soc[{i}]"
        )

    charge = lambda t: v["charge"][t] if battery else 0
    discharge = lambda t: v["discharge"][t] if battery else 0

    c = {
        "net": m.addConstrs(
            (
                v["q"][t]
                - v["load"][t]
                - charge(t)
                + v["pv"][t]
                + discharge(t) == 0
                for t in T
            ),
            name=f"net[{i}]"
        ),

        "deviation": m.addConstrs(
            (
                v["load"][t]
                - ref[t]
                - v["up"][t]
                + v["down"][t] == 0
                for t in T
            ),
            name=f"deviation[{i}]"
        ),

        "load_lower": m.addConstrs(
            (v["load"][t] >= lo[t] for t in T),
            name=f"load_lower[{i}]"
        ),

        "load_upper": m.addConstrs(
            (v["load"][t] <= hi[t] for t in T),
            name=f"load_upper[{i}]"
        ),

        "pv_available": m.addConstrs(
            (v["pv"][t] <= available[t] for t in T),
            name=f"pv_limit[{i}]"
        ),
    }

    if d.min_daily_energy_kWh is not None:
        c["daily_min"] = m.addConstr(
            gp.quicksum(v["load"][t] for t in T)
            >= d.min_daily_energy_kWh,
            name=f"daily_min[{i}]"
        )

    if battery:
        initial = d.battery_initial_soc_kWh
        eta_c = d.battery_charging_efficiency
        eta_d = d.battery_discharging_efficiency

        c["soc"] = m.addConstrs(
            (
                v["soc"][t]
                - (v["soc"][t-1] if t else initial)
                - eta_c * v["charge"][t]
                + v["discharge"][t] / eta_d == 0
                for t in T
            ),
            name=f"soc_balance[{i}]"
        )

        if end_soc == "initial":
            # Same convention as Q1:
            # final SoC must be at least initial SoC.
            c["soc_end"] = m.addConstr(
                v["soc"][len(T)-1] >= initial,
                name=f"soc_end[{i}]"
            )
        elif end_soc != "free":
            raise ValueError(
                "end_soc must be 'initial' or 'free'"
            )

    # Physical cost: load disutility + PV generation cost
    D = (
        d.linear_disutility
        * gp.quicksum(
            v["up"][t] + v["down"][t] for t in T
        )
        + d.pv_marginal_cost
        * gp.quicksum(v["pv"][t] for t in T)
    )

    return {"v": v, "c": c, "D": D}


# ============================================================
# CENTRAL BENCHMARK: MODEL C FROM QUESTION 1
# ============================================================

def solve_central(com, *, discount=None, cap=None,
                  verbose=False):
    """Solve the centralized community benchmark C."""

    ct = com.contract

    beta = (
        ct.tariff_discount_ratio
        if discount is None else float(discount)
    )

    T = range(len(com.energy_price))
    p = np.asarray(com.energy_price, float)

    cap = np.broadcast_to(
        np.asarray(
            ct.capacity_limit_kW if cap is None else cap,
            float
        ),
        (len(T),)
    )

    m = gp.Model("central_C")
    m.Params.OutputFlag = int(verbose)

    blocks = {
        d.consumer_id: add_member_block(m, d)
        for d in com.members
    }

    imp, exp, exceed = (
        m.addVars(T, name=name)
        for name in ("P_imp", "P_exp", "E")
    )

    balance = m.addConstrs(
        (
            imp[t] - exp[t]
            == gp.quicksum(
                b["v"]["q"][t] for b in blocks.values()
            )
            for t in T
        ),
        name="balance"
    )

    cap_con = m.addConstrs(
        (imp[t] - exceed[t] <= cap[t] for t in T),
        name="cap"
    )

    a = p + (1-beta) * com.import_tariff
    b = p - (1-beta) * com.export_tariff

    bill = gp.quicksum(
        a[t] * imp[t]
        - b[t] * exp[t]
        + ct.penalty_DKK_per_kWh * exceed[t]
        for t in T
    )

    m.setObjective(
        bill + gp.quicksum(
            block["D"] for block in blocks.values()
        ),
        GRB.MINIMIZE
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Central C failed with status {m.Status}"
        )

    hourly = pd.DataFrame({
        "hour": list(T),
        "lambda": [balance[t].Pi for t in T],
        "P_imp": [imp[t].X for t in T],
        "P_exp": [exp[t].X for t in T],
        "E": [exceed[t].X for t in T],
        "cap": cap,
        "mu": [-cap_con[t].Pi for t in T],
    })

    return {
        "cost": m.ObjVal,
        "grid_cost": bill.getValue(),
        "hourly": hourly,
        "q": {
            i: np.array([
                block["v"]["q"][t].X for t in T
            ])
            for i, block in blocks.items()
        },
        "D": {
            i: block["D"].getValue()
            for i, block in blocks.items()
        }
    }


# ============================================================
# STEP 1: INDIVIDUAL BEST RESPONSE
# ============================================================

def best_response(d, prices):
    """Solve one member's LP at fixed internal prices x."""

    x = np.asarray(prices, float)
    T = range(len(d.energy_price))

    if x.shape != (len(T),):
        raise ValueError("Expected one price per hour")

    m = gp.Model(f"response_{d.consumer_id}")
    m.Params.OutputFlag = 0

    block = add_member_block(m, d)

    objective = (
        block["D"]
        + gp.quicksum(
            float(x[t]) * block["v"]["q"][t]
            for t in T
        )
    )

    m.setObjective(objective, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Best response failed: {d.consumer_id}, "
            f"status {m.Status}"
        )

    return {
        "cost": m.ObjVal,
        "model": m,
        "block": block,
        "load": np.array([
            block["v"]["load"][t].X for t in T
        ]),
        "q": np.array([
            block["v"]["q"][t].X for t in T
        ]),
    }


# ============================================================
# STEP 2: KKT AND COMPLEMENTARITY
# ============================================================

def _complementarity(m, multiplier, slack_expr, name):
    """Enforce multiplier >= 0, slack >= 0 and product = 0."""

    slack = m.addVar(
        lb=0.0,
        name=f"slack[{name}]"
    )

    m.addConstr(
        slack == slack_expr,
        name=f"define_slack[{name}]"
    )

    m.addSOS(
        GRB.SOS_TYPE1,
        [multiplier, slack]
    )

    return slack


def add_kkt(m, d, block, x):
    """Add all KKT conditions of one member's linear program.

    Includes:
      - Original physical constraints
      - Dual feasibility
      - Stationarity
      - Complementarity using SOS1
      - Linear dual objective for strong duality

    Handles both explicit constraints and variable bounds.
    """

    m.update()

    i = d.consumer_id

    primal = [
        v
        for family in block["v"].values()
        for v in family.values()
    ]

    rows = []

    for name, con in block["c"].items():
        if isinstance(con, gp.tupledict):
            for key, value in con.items():
                rows.append((f"{name},{key}", value))
        else:
            rows.append((name, con))

    # Gradient of the member's objective
    gradient = {
        v.index: gp.LinExpr()
        for v in primal
    }

    D = block["D"]

    for k in range(D.size()):
        v = D.getVar(k)
        gradient[v.index] += D.getCoeff(k)

    for t in range(len(d.energy_price)):
        gradient[block["v"]["q"][t].index] += x[t]

    # Dual objective (linear expression)
    phi = gp.LinExpr()

    pairs = []
    stations = []

    # Dual variables of physical constraints
    for name, con in rows:

        lhs = m.getRow(con)
        rhs = float(con.RHS)
        sense = con.Sense

        if sense == "=":
            mu = m.addVar(
                lb=-GRB.INFINITY,
                name=f"pi[{i},{name}]"
            )
            sign = 1
            phi += -rhs * mu

        elif sense == "<":
            mu = m.addVar(
                lb=0,
                name=f"mu_le[{i},{name}]"
            )
            sign = 1
            phi += -rhs * mu

            pairs.append((
                mu,
                _complementarity(
                    m, mu, rhs-lhs, f"{i},{name}"
                )
            ))

        elif sense == ">":
            mu = m.addVar(
                lb=0,
                name=f"mu_ge[{i},{name}]"
            )
            sign = -1
            phi += rhs * mu

            pairs.append((
                mu,
                _complementarity(
                    m, mu, lhs-rhs, f"{i},{name}"
                )
            ))

        else:
            raise ValueError(sense)

        for k in range(lhs.size()):
            v = lhs.getVar(k)

            gradient[v.index] += (
                sign * lhs.getCoeff(k) * mu
            )

    # Dual variables of implicit variable bounds
    for v in primal:

        if v.LB > -GRB.INFINITY / 2:
            mu = m.addVar(
                lb=0,
                name=f"mu_lb[{i},{v.VarName}]"
            )

            gradient[v.index] += -mu
            phi += float(v.LB) * mu

            pairs.append((
                mu,
                _complementarity(
                    m,
                    mu,
                    v - float(v.LB),
                    f"{i},lb,{v.index}"
                )
            ))

        if v.UB < GRB.INFINITY / 2:
            mu = m.addVar(
                lb=0,
                name=f"mu_ub[{i},{v.VarName}]"
            )

            gradient[v.index] += mu
            phi += -float(v.UB) * mu

            pairs.append((
                mu,
                _complementarity(
                    m,
                    mu,
                    float(v.UB) - v,
                    f"{i},ub,{v.index}"
                )
            ))

        # Stationarity for every primal variable
        stations.append(
            m.addConstr(
                gradient[v.index] == 0,
                name=f"stationarity[{i},{v.index}]"
            )
        )

    return {
        "phi": phi,
        "pairs": pairs,
        "stations": stations,
    }


# ============================================================
# STEP 2 + 4: PARAMETRIZED DYNAMIC PRICING MODEL
# ============================================================

def build_dp(
    com,
    variant="cap",
    R=None,
    *,
    discount=None,
    cap=None,
    price_bound=None,
    min_relax=False,
    verbose=False
):
    """Build optimistic bilevel MILP using KKT.

    Variants:
        cap         : basic contract only
        bb          : budget balance
        bb_ir       : budget balance + individual rationality
        bb_ir_relax : relaxed individual rationality

    price_bound=None: no additional price restriction.
    """

    allowed = ("cap", "bb", "bb_ir", "bb_ir_relax")

    if variant not in allowed:
        raise ValueError("Unknown variant")

    if variant in ("bb_ir", "bb_ir_relax") and R is None:
        raise ValueError(
            "Individual rationality requires the R benchmark"
        )

    ct = com.contract

    beta = (
        ct.tariff_discount_ratio
        if discount is None else float(discount)
    )

    T = range(len(com.energy_price))

    cap = np.broadcast_to(
        np.asarray(
            ct.capacity_limit_kW if cap is None else cap,
            float
        ),
        (len(T),)
    )

    p = np.asarray(com.energy_price, float)

    m = gp.Model(f"DP_{variant}")
    m.Params.OutputFlag = int(verbose)

    bound = (
        GRB.INFINITY
        if price_bound is None
        else float(price_bound)
    )

    if bound <= 0:
        raise ValueError("price_bound must be positive")

    # Upper-level decision: hourly internal prices
    x = m.addVars(
        T,
        lb=-bound,
        ub=bound,
        name="x"
    )

    # Members' physical constraints
    blocks = {
        d.consumer_id: add_member_block(m, d)
        for d in com.members
    }

    # Community meter
    imp, exp, exceed = (
        m.addVars(T, name=name)
        for name in ("P_imp", "P_exp", "E")
    )

    m.addConstrs(
        (
            imp[t] - exp[t]
            == gp.quicksum(
                b["v"]["q"][t]
                for b in blocks.values()
            )
            for t in T
        ),
        name="community_balance"
    )

    m.addConstrs(
        (imp[t]-exceed[t] <= cap[t] for t in T),
        name="DSO_cap"
    )

    # Same external grid-cost formulation as Q1
    a = p + (1-beta) * com.import_tariff
    b = p - (1-beta) * com.export_tariff

    grid = gp.quicksum(
        a[t] * imp[t]
        - b[t] * exp[t]
        + ct.penalty_DKK_per_kWh * exceed[t]
        for t in T
    )

    D = {
        i: blk["D"]
        for i, blk in blocks.items()
    }

    # Total community welfare cost
    welfare = grid + gp.quicksum(D.values())

    # Replace each lower-level LP with KKT
    kkt = {
        d.consumer_id: add_kkt(
            m, d, blocks[d.consumer_id], x
        )
        for d in com.members
    }

    # Strong duality: member cost and payment
    phi = {
        i: kkt[i]["phi"]
        for i in blocks
    }

    pay = {
        i: phi[i]-D[i]
        for i in blocks
    }

    # Budget balance
    if variant != "cap":
        m.addConstr(
            gp.quicksum(pay.values()) == grid,
            name="budget_balance"
        )

    # Individual rationality
    relax = {}

    if variant in ("bb_ir", "bb_ir_relax"):
        for i in blocks:

            if variant == "bb_ir_relax":
                relax[i] = m.addVar(
                    lb=0,
                    name=f"IR_relax[{i}]"
                )

            m.addConstr(
                phi[i] <= float(R[i])
                + (relax[i] if i in relax else 0),
                name=f"IR[{i}]"
            )

    if min_relax:
        if variant != "bb_ir_relax":
            raise ValueError(
                "min_relax requires bb_ir_relax"
            )
        m.setObjective(
            gp.quicksum(relax.values()),
            GRB.MINIMIZE
        )
    else:
        m.setObjective(welfare, GRB.MINIMIZE)

    m.update()

    return {
        "model": m,
        "variant": variant,
        "x": x,
        "blocks": blocks,
        "kkt": kkt,
        "imp": imp,
        "exp": exp,
        "exceed": exceed,
        "cap": cap,
        "grid": grid,
        "D": D,
        "phi": phi,
        "pay": pay,
        "relax": relax,
        "welfare": welfare,
    }


# ============================================================
# SOLVE MILP AND EXTRACT RESULTS
# ============================================================

def solve_dp(bundle, *, time_limit=300, verbose=False):
    """Solve DP and return results with numerical KKT checks."""

    m = bundle["model"]

    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = 1e-4

    m.optimize()

    status = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }.get(m.Status, str(m.Status))

    out = {
        "status": status,
        "variant": bundle["variant"],
        "runtime_s": m.Runtime,
        "mip_gap": float(m.MIPGap)
        if m.SolCount else np.nan,
        "n_vars": m.NumVars,
        "n_sos": m.NumSOS,
        "has_solution": m.SolCount > 0,
    }

    if not out["has_solution"]:
        return out

    T = range(len(bundle["x"]))

    x = np.array([
        bundle["x"][t].X for t in T
    ])

    hourly = pd.DataFrame({
        "hour": list(T),
        "x": x,
        "P_imp": [
            bundle["imp"][t].X for t in T
        ],
        "P_exp": [
            bundle["exp"][t].X for t in T
        ],
        "E": [
            bundle["exceed"][t].X for t in T
        ],
        "cap": bundle["cap"],
    })

    member_rows = []

    max_pay_gap = 0.0
    max_comp = 0.0
    max_stat = 0.0

    for i, blk in bundle["blocks"].items():

        q = np.array([
            blk["v"]["q"][t].X for t in T
        ])

        cost = bundle["phi"][i].getValue()
        D = bundle["D"][i].getValue()
        payment = bundle["pay"][i].getValue()

        max_pay_gap = max(
            max_pay_gap,
            abs(payment - x @ q)
        )

        for a, b in bundle["kkt"][i]["pairs"]:
            max_comp = max(
                max_comp, abs(a.X * b.X)
            )

        for c in bundle["kkt"][i]["stations"]:
            max_stat = max(
                max_stat, abs(c.Slack)
            )

        member_rows.append({
            "consumer_id": i,
            "cost_DKK": cost,
            "payment_DKK": payment,
            "physical_cost_DKK": D,
            "IR_relax_DKK":
                bundle["relax"][i].X
                if i in bundle["relax"] else 0,
            "net_kWh": q.sum(),
            **{
                f"q_{t}": q[t]
                for t in T
            }
        })

    members = pd.DataFrame(member_rows)
    grid_value = bundle["grid"].getValue()

    out.update({
        "objective_DKK":
            bundle["welfare"].getValue(),
        "solver_objective": m.ObjVal,
        "grid_cost_DKK": grid_value,
        "budget_surplus_DKK":
            members.payment_DKK.sum() - grid_value,
        "cap_exceedance_kWh": hourly.E.sum(),
        "peak_import_kW": hourly.P_imp.max(),
        "members": members,
        "hourly": hourly,
        "max_payment_gap": max_pay_gap,
        "max_stationarity": max_stat,
        "max_complementarity": max_comp,
        "total_IR_relax_DKK":
            members.IR_relax_DKK.sum(),
    })

    return out


# ============================================================
# STEP 3: INDIFFERENCE ANALYSIS
# ============================================================

def indifference(com, prices, tol=1e-7):
    """Identify hours with multiple optimal member responses.

    For fixed prices, calculate the minimum and maximum
    hourly load and net exchange while preserving optimal cost.
    """

    rows = []

    for d in com.members:

        r = best_response(d, prices)

        m = r["model"]
        b = r["block"]

        m.addConstr(
            m.getObjective() <= r["cost"] + tol,
            name="optimal_face"
        )

        for t in range(len(prices)):

            rec = {
                "consumer_id": d.consumer_id,
                "hour": t,
            }

            for name in ("load", "q"):

                var = b["v"][name][t]

                m.setObjective(var, GRB.MINIMIZE)
                m.optimize()

                if m.Status != GRB.OPTIMAL:
                    raise RuntimeError(
                        "Indifference min failed"
                    )

                lo = var.X

                m.setObjective(var, GRB.MAXIMIZE)
                m.optimize()

                if m.Status != GRB.OPTIMAL:
                    raise RuntimeError(
                        "Indifference max failed"
                    )

                rec[f"{name}_range_kWh"] = var.X-lo

            if (
                rec["load_range_kWh"] > 1e-4
                or rec["q_range_kWh"] > 1e-4
            ):
                rows.append(rec)

    return pd.DataFrame(
        rows,
        columns=[
            "consumer_id",
            "hour",
            "load_range_kWh",
            "q_range_kWh",
        ]
    )
