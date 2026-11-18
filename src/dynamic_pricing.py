"""Question 2: dynamic pricing (model DP) for the energy community.

The members' constraints are the ones of Question 1 (add_member_block in community_model.py),
so they are written in one place only.

    best_response   one member's LP at fixed internal prices x_t (the lower level of DP)
"""
from __future__ import annotations

import gurobipy as gp
import numpy as np
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