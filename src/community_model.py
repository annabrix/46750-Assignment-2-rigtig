"""One parametrized model of the community, implemented with gurobipy: R, I and C are switches of it.

    model = CommunityModel(com.members)
    model.build(pooled=True, discount=0.75, cap_kW=cap, penalty=10)    # C: one meter, one cap
    res = model.solve()

"""
from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


def add_member_block(m: gp.Model, d, tag: str, end_soc: str = "initial") -> dict:
    """Add the variables and constraints of one member to model ``m``.

    Returns a dict with the member's variables, its constraints, its net consumption ``q`` (kWh/h, free:
    positive = takes energy, negative = gives energy) and its cost terms ``disutility`` and ``pv_cost`` (DKK).
    """
    T = range(len(d.energy_price))
    ref = np.asarray(d.reference_load, dtype=float)
    lo = np.asarray(d.load_lower_kWh, dtype=float)
    hi = np.asarray(d.load_upper_kWh, dtype=float)
    pv_av = np.asarray(d.pv_available, dtype=float) if (d.pv_max_kW or 0) > 0 else np.zeros(len(T))
    has_bat = d.battery_capacity_kWh is not None and d.battery_capacity_kWh > 0

    v, c = {}, {}
    v["load"] = m.addVars(T, name=f"load[{tag}]")
    v["dev_up"] = m.addVars(T, name=f"dev_up[{tag}]")
    v["dev_dn"] = m.addVars(T, name=f"dev_dn[{tag}]")
    v["pv"] = m.addVars(T, name=f"pv[{tag}]")
    v["q"] = m.addVars(T, lb=-GRB.INFINITY, name=f"q[{tag}]")
    if has_bat:
        v["charge"] = m.addVars(T, ub=d.battery_max_charge_kW, name=f"charge[{tag}]")
        v["discharge"] = m.addVars(T, ub=d.battery_max_discharge_kW, name=f"discharge[{tag}]")
        v["soc"] = m.addVars(T, ub=d.battery_capacity_kWh, name=f"soc[{tag}]")
    ch = (lambda t: v["charge"][t]) if has_bat else (lambda t: 0.0)
    dis = (lambda t: v["discharge"][t]) if has_bat else (lambda t: 0.0)

    # net consumption: load + charge - PV - discharge
    c["net"] = m.addConstrs((v["q"][t] - v["load"][t] - ch(t) + v["pv"][t] + dis(t) == 0 for t in T),
                            name=f"net[{tag}]")
    c["deviation"] = m.addConstrs((v["load"][t] - ref[t] - v["dev_up"][t] + v["dev_dn"][t] == 0 for t in T),
                                  name=f"deviation[{tag}]")
    c["load_lower"] = m.addConstrs((v["load"][t] >= lo[t] for t in T), name=f"load_lower[{tag}]")
    c["load_upper"] = m.addConstrs((v["load"][t] <= hi[t] for t in T), name=f"load_upper[{tag}]")
    c["pv_available"] = m.addConstrs((v["pv"][t] <= pv_av[t] for t in T), name=f"pv_available[{tag}]")
    if d.min_daily_energy_kWh is not None:
        c["daily_min"] = m.addConstr(gp.quicksum(v["load"][t] for t in T) >= d.min_daily_energy_kWh,
                                     name=f"daily_min[{tag}]")
    if has_bat:
        s0 = d.battery_initial_soc_kWh
        eta_c, eta_d = d.battery_charging_efficiency, d.battery_discharging_efficiency
        c["soc"] = m.addConstrs(
            (v["soc"][t] - (v["soc"][t - 1] if t > 0 else s0)
             - eta_c * v["charge"][t] + v["discharge"][t] / eta_d == 0 for t in T), name=f"soc[{tag}]")
        if end_soc == "initial":
            c["soc_end"] = m.addConstr(v["soc"][T[-1]] >= s0, name=f"soc_end[{tag}]")

    disutility = d.linear_disutility * gp.quicksum(v["dev_up"][t] + v["dev_dn"][t] for t in T)
    pv_cost = d.pv_marginal_cost * gp.quicksum(v["pv"][t] for t in T)
    return {"var": v, "con": c, "q": v["q"], "disutility": disutility, "pv_cost": pv_cost}


@dataclass
class CommunityResults:
    """Solution of one run of ``CommunityModel``."""

    status: str
    objective: float                        # DKK: disutility + PV cost + grid bill(s)
    hourly: pd.DataFrame                    # one row per hour: P_imp, P_exp, exceedance, cap, lambda, mu
    members: pd.DataFrame                   # one row per member: costs and energies
    q: pd.DataFrame                         # net consumption, hours x members (kWh/h)
    meta: dict = field(default_factory=dict)


class CommunityModel:
    """Members R / I / C as switches of one LP (see the module docstring)."""

    def __init__(self, members, name: str = "community", verbose: bool = False):
        self.members = list(members)
        self.ids = [d.consumer_id for d in self.members]
        self.T = range(len(self.members[0].energy_price))
        self.p = np.asarray(self.members[0].energy_price, dtype=float)
        self.tau_imp = self.members[0].import_tariff
        self.tau_exp = self.members[0].export_tariff
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.blocks: dict[str, dict] = {}
        self.var: dict = {}
        self.con: dict = {}

    def build(self, pooled: bool = True, discount: float = 0.0, cap_kW=None, penalty: float = 0.0,
              end_soc: str = "initial") -> "CommunityModel":
        m, T, p = self.m, self.T, self.p
        self.pooled, self.discount, self.penalty = pooled, discount, penalty
        ti, te = (1 - discount) * self.tau_imp, (1 - discount) * self.tau_exp
        for d in self.members:
            self.blocks[d.consumer_id] = add_member_block(m, d, d.consumer_id, end_soc)
        member_cost = gp.quicksum(b["disutility"] + b["pv_cost"] for b in self.blocks.values())

        if pooled:   # model C: one meter, one bill, hourly balance behind the meter
            P_imp = self.var["P_imp"] = m.addVars(T, name="P_imp")
            P_exp = self.var["P_exp"] = m.addVars(T, name="P_exp")
            # written as P_imp - P_exp - sum_i q_i = 0, so that its dual is +lambda_t (DKK/kWh)
            self.con["balance"] = m.addConstrs(
                (P_imp[t] - P_exp[t] - gp.quicksum(b["q"][t] for b in self.blocks.values()) == 0 for t in T),
                name="balance")
            bill = gp.quicksum((p[t] + ti) * P_imp[t] - (p[t] - te) * P_exp[t] for t in T)
            if cap_kW is not None:
                cap = np.broadcast_to(np.asarray(cap_kW, dtype=float), len(T))
                E = self.var["E"] = m.addVars(T, name="E")
                self.con["cap"] = m.addConstrs((P_imp[t] - E[t] <= cap[t] for t in T), name="cap")
                bill += penalty * gp.quicksum(E[t] for t in T)
                self.cap = cap
            else:
                self.cap = None
        else:        # models R and I: every member behind its own meter, billed on its own flows
            bill = 0
            self.var["imp"], self.var["exp"], self.var["e"], self.con["cap_i"] = {}, {}, {}, {}
            for i, b in self.blocks.items():
                imp = self.var["imp"][i] = m.addVars(T, name=f"imp[{i}]")
                exp = self.var["exp"][i] = m.addVars(T, name=f"exp[{i}]")
                m.addConstrs((b["q"][t] - imp[t] + exp[t] == 0 for t in T), name=f"meter[{i}]")
                bill += gp.quicksum((p[t] + ti) * imp[t] - (p[t] - te) * exp[t] for t in T)
                if cap_kW is not None:
                    cap_i = np.broadcast_to(np.asarray(cap_kW[i], dtype=float), len(T))
                    e = self.var["e"][i] = m.addVars(T, name=f"e[{i}]")
                    self.con["cap_i"][i] = m.addConstrs((imp[t] - e[t] <= cap_i[t] for t in T), name=f"cap[{i}]")
                    bill += penalty * gp.quicksum(e[t] for t in T)
            self.cap = None if cap_kW is None else np.asarray(sum(np.broadcast_to(np.asarray(cap_kW[i], float), len(T))
                                                                   for i in self.ids))
        self.bill = bill
        m.setObjective(member_cost + bill, GRB.MINIMIZE)
        m.update()
        return self

    def solve(self) -> CommunityResults:
        m, T = self.m, list(self.T)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"CommunityModel ended with status {m.Status}")

        q = pd.DataFrame({i: [b["q"][t].X for t in T] for i, b in self.blocks.items()}, index=pd.Index(T, name="hour"))
        rows = []
        for d in self.members:
            i = d.consumer_id
            b = self.blocks[i]
            row = {"consumer_id": i, "archetype": getattr(d, "archetype", ""),
                   "disutility_DKK": b["disutility"].getValue(), "pv_cost_DKK": b["pv_cost"].getValue(),
                   "load_kWh": sum(b["var"]["load"][t].X for t in T), "net_kWh": q[i].sum(),
                   "daily_min_dual": b["con"]["daily_min"].Pi if "daily_min" in b["con"] else np.nan}
            if not self.pooled:
                imp = np.array([self.var["imp"][i][t].X for t in T])
                exp = np.array([self.var["exp"][i][t].X for t in T])
                e = np.array([self.var["e"][i][t].X for t in T]) if self.var["e"] else np.zeros(len(T))
                row.update(import_kWh=imp.sum(), export_kWh=exp.sum(), exceed_kWh=e.sum())
            rows.append(row)
        members = pd.DataFrame(rows)

        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = self.p
        if self.pooled:
            hourly["P_imp"] = [self.var["P_imp"][t].X for t in T]
            hourly["P_exp"] = [self.var["P_exp"][t].X for t in T]
            hourly["lambda"] = [self.con["balance"][t].Pi for t in T]          # internal price, DKK/kWh
            if self.cap is not None:
                hourly["E"] = [self.var["E"][t].X for t in T]
                hourly["mu"] = [max(0.0, -self.con["cap"][t].Pi) for t in T]         # cap multiplier >= 0, DKK/kWh
        else:
            hourly["P_imp"] = sum(np.array([self.var["imp"][i][t].X for t in T]) for i in self.ids)
            hourly["P_exp"] = sum(np.array([self.var["exp"][i][t].X for t in T]) for i in self.ids)
        if self.cap is not None:
            hourly["cap"] = self.cap

        return CommunityResults(status="OPTIMAL", objective=m.ObjVal, hourly=hourly, members=members, q=q,
                                meta={"bill_DKK": self.bill.getValue() if hasattr(self.bill, "getValue") else 0.0,
                                      "n_vars": m.NumVars, "n_constrs": m.NumConstrs,
                                      "runtime_s": m.Runtime})
