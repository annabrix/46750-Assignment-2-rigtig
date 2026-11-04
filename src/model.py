"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

"""



from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self, discount: float = 0.0, cap_kW=None, penalty: float = 0.0,
              tariff_over: str = "discounted", end_soc: str = "initial") -> "FlexibleConsumerModel":
        """Consumer problem against prices and tariffs, as one parametrized model.

        Defaults give model R (no contract, full tariffs). The switches turn it into model I:
          discount     beta^DSO in [0, 1], applied to both tariffs
          cap_kW       None (no cap) or the consumer's cap kappa_i * P_bar (float or 24 values)
          penalty      alpha^DSO, DKK per kWh imported above the cap
          tariff_over  tariff paid on above-cap kWh: "none", "discounted" or "full"
          end_soc      "initial" (SoC_24 >= SoC_0) or "free"
        """
        d, m, T = self.data, self.m, self.T
        p = d.energy_price
        tau_imp = (1 - discount) * d.import_tariff
        tau_exp = (1 - discount) * d.export_tariff
        tau_over = {"none": 0.0, "discounted": tau_imp, "full": d.import_tariff}[tariff_over]

        # --- parameters that may be absent for a member (no PV, no battery) ------------
        ref = d.reference_load
        delta = getattr(d, "max_hourly_deviation_kWh", None)
        lo = getattr(d, "load_lower_kWh", None)
        hi = getattr(d, "load_upper_kWh", None)
        if lo is None:  # fall back on the bounds of the instructions
            lo = np.maximum(d.load_min_kWh, ref - delta) if delta is not None else np.full(len(T), d.load_min_kWh)
        if hi is None:
            hi = np.minimum(d.load_max_kWh, ref + delta) if delta is not None else np.full(len(T), d.load_max_kWh)
        lo, hi = np.broadcast_to(lo, len(T)), np.broadcast_to(hi, len(T))
        pv_av = d.pv_available if (d.pv_max_kW or 0) > 0 else np.zeros(len(T))
        has_bat = d.battery_capacity_kWh is not None and d.battery_capacity_kWh > 0

        # --- variables (names "import", "export", "load", "pv" feed src/plotting.py) -------
        self.var["load"] = m.addVars(T, name="load")
        self.var["dev_up"] = m.addVars(T, name="dev_up")
        self.var["dev_dn"] = m.addVars(T, name="dev_dn")
        self.var["pv"] = m.addVars(T, name="pv")
        self.var["import"] = m.addVars(T, name="import")
        self.var["export"] = m.addVars(T, name="export")
        if cap_kW is not None:
            self.var["exceed"] = m.addVars(T, name="exceed")
        if has_bat:
            self.var["charge"] = m.addVars(T, ub=d.battery_max_charge_kW, name="charge")
            self.var["discharge"] = m.addVars(T, ub=d.battery_max_discharge_kW, name="discharge")
            self.var["soc"] = m.addVars(T, ub=d.battery_capacity_kWh, name="soc")
        v = self.var
        ch = (lambda t: v["charge"][t]) if has_bat else (lambda t: 0.0)
        dis = (lambda t: v["discharge"][t]) if has_bat else (lambda t: 0.0)
        exc = (lambda t: v["exceed"][t]) if cap_kW is not None else (lambda t: 0.0)

        # --- objective: energy + tariffs + penalty - export revenue + disutility + PV cost --
        m.setObjective(gp.quicksum(
            (p[t] + tau_imp) * (v["import"][t] - exc(t))
            + (p[t] + tau_over + penalty) * exc(t)
            - (p[t] - tau_exp) * v["export"][t]
            + d.linear_disutility * (v["dev_up"][t] + v["dev_dn"][t])
            + d.pv_marginal_cost * v["pv"][t]
            for t in T), GRB.MINIMIZE)

        # --- constraints (explicit where a dual is useful later) -----------------------------
        self.con["balance"] = m.addConstrs(
            (v["load"][t] + ch(t) + v["export"][t] - v["pv"][t] - dis(t) - v["import"][t] == 0 for t in T),
            name="balance")
        self.con["deviation"] = m.addConstrs(
            (v["load"][t] - ref[t] - v["dev_up"][t] + v["dev_dn"][t] == 0 for t in T), name="deviation")
        self.con["load_lower"] = m.addConstrs((v["load"][t] >= lo[t] for t in T), name="load_lower")
        self.con["load_upper"] = m.addConstrs((v["load"][t] <= hi[t] for t in T), name="load_upper")
        self.con["pv_available"] = m.addConstrs((v["pv"][t] <= pv_av[t] for t in T), name="pv_available")
        if d.min_daily_energy_kWh is not None:
            self.con["daily_min"] = m.addConstr(
                gp.quicksum(v["load"][t] for t in T) >= d.min_daily_energy_kWh, name="daily_min")
        if cap_kW is not None:
            cap = np.broadcast_to(cap_kW, len(T))
            # import - exceed <= cap: the import below the cap; exceed >= 0 takes the rest
            self.con["cap"] = m.addConstrs(
                (v["import"][t] - v["exceed"][t] <= cap[t] for t in T), name="cap")
        if has_bat:
            s0, eta_c, eta_d = d.battery_initial_soc_kWh, d.battery_charging_efficiency, d.battery_discharging_efficiency
            self.con["soc"] = m.addConstrs(
                (v["soc"][t] - (v["soc"][t - 1] if t > 0 else s0)
                 - eta_c * v["charge"][t] + v["discharge"][t] / eta_d == 0 for t in T), name="soc")
            if end_soc == "initial":
                self.con["soc_end"] = m.addConstr(v["soc"][T[-1]] >= s0, name="soc_end")

        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
