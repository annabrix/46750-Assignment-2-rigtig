"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import FlexibleConsumerModel
from src.community_model import CommunityModel
from src.dynamic_pricing import build_dp, solve_dp, solution_values
from src.stochastic_pricing import draw_scenarios, evaluate_prices, StochasticDP

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def run_reference(com, out_dir: Path) -> None:
    """Model R: every member alone, full tariffs, no contract.
    """
    rows, P_imp, P_exp = [], np.zeros(24), np.zeros(24)
    for d in com.members:
        r = FlexibleConsumerModel(d, name=f"R_{d.consumer_id}").build().solve()
        h = r.hourly
        bill = ((d.energy_price + d.import_tariff) * h["import"]
                - (d.energy_price - d.export_tariff) * h["export"]).sum()     # energy + tariffs - export revenue
        rows.append({
            "consumer_id": d.consumer_id, "archetype": d.archetype,
            "import_kWh": h["import"].sum(), "export_kWh": h["export"].sum(),
            "bill_DKK": bill, "pv_cost_DKK": d.pv_marginal_cost * h["pv"].sum(),
            "objective_DKK": r.objective, "daily_min_dual": r.duals.get("daily_min"),
        })
        P_imp += h["import"].to_numpy()
        P_exp += h["export"].to_numpy()

    members = pd.DataFrame(rows)
    cap = np.asarray(com.contract.capacity_limit_kW, dtype=float)
    over = np.maximum(P_imp - cap, 0.0)

    print("\nModel R, per member:")
    print(members.round(2).to_string(index=False))

    ref = com.contract.reference_model_R
    print("\nModel R, aggregated            ours      file")
    print(f"  daily import (kWh)       {P_imp.sum():8.2f}  {ref.get('daily_import_kWh')}")
    print(f"  peak import (kW)         {P_imp.max():8.2f}  {ref.get('peak_import_kW')}")
    print(f"  exceedance (kWh)         {over.sum():8.2f}  {ref.get('exceedance_kWh')}")
    print(f"  hours above cap          {int((over > 1e-6).sum()):8d}  9 (instructions)")
    print(f"  cost (DKK)               {members.objective_DKK.sum():8.2f}  {ref.get('cost_DKK')}")

    out_dir.mkdir(parents=True, exist_ok=True)
    members.to_csv(out_dir / "R_members.csv", index=False)
    pd.DataFrame({"P_imp_kW": P_imp, "P_exp_kW": P_exp, "cap_kW": cap}).to_csv(
        out_dir / "R_hourly.csv", index_label="hour")


def run_individual(com, out_dir: Path) -> None:
    """Model I """
    ct = com.contract
    beta, alpha = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh
    cap = np.asarray(ct.capacity_limit_kW, dtype=float)          # community cap, kW
    rows, P_R, P_I = [], np.zeros(24), np.zeros(24)
    for d in com.members:
        cap_i = np.asarray(ct.individual_capacity_limit_kW[d.consumer_id], dtype=float)
        rR = FlexibleConsumerModel(d, name=f"R_{d.consumer_id}").build().solve()
        rI = FlexibleConsumerModel(d, name=f"I_{d.consumer_id}").build(
            discount=beta, cap_kW=cap_i, penalty=alpha, tariff_over="discounted").solve()
        h = rI.hourly
        rows.append({
            "consumer_id": d.consumer_id, "archetype": d.archetype,
            "cap_i_kW": cap_i[0],
            "import_R_kWh": rR.hourly["import"].sum(), "import_I_kWh": h["import"].sum(),
            "exceed_I_kWh": h["exceed"].sum(), "penalty_I_DKK": alpha * h["exceed"].sum(),
            "cost_R_DKK": rR.objective, "cost_I_DKK": rI.objective,
            "gain_I_vs_R_DKK": rR.objective - rI.objective,
        })
        P_R += rR.hourly["import"].to_numpy()
        P_I += h["import"].to_numpy()

    members = pd.DataFrame(rows)
    over_R, over_I = np.maximum(P_R - cap, 0.0), np.maximum(P_I - cap, 0.0)

    print("\nModel I, per member (gain > 0: signing alone is better than no contract):")
    print(members.round(2).to_string(index=False))

    print("\nCongestion at the connection point        R          I")
    print(f"  daily import (kWh)                 {P_R.sum():8.2f}  {P_I.sum():8.2f}")
    print(f"  peak import (kW)                   {P_R.max():8.2f}  {P_I.max():8.2f}")
    print(f"  hours above community cap         {int((over_R > 1e-6).sum()):8d}  {int((over_I > 1e-6).sum()):8d}")
    print(f"  exceedance of community cap (kWh)  {over_R.sum():8.2f}  {over_I.sum():8.2f}")
    print(f"  sum of individual exceedances (kWh)        -  {members.exceed_I_kWh.sum():8.2f}")
    print(f"  penalties paid (DKK)                       -  {members.penalty_I_DKK.sum():8.2f}")
    print(f"  total cost of the members (DKK)    {members.cost_R_DKK.sum():8.2f}  {members.cost_I_DKK.sum():8.2f}")
    print(f"  members better off in I than in R: {int((members.gain_I_vs_R_DKK > 1e-6).sum())} of {len(members)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    members.to_csv(out_dir / "I_members.csv", index=False)
    pd.DataFrame({"P_R_kW": P_R, "P_I_kW": P_I, "cap_kW": cap}).to_csv(
        out_dir / "I_hourly.csv", index_label="hour")
    
def run_central(com, out_dir: Path) -> None:
    """Model C: """
    ct = com.contract
    beta, alpha = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh
    cap = np.asarray(ct.capacity_limit_kW, dtype=float)
    cap_i = {i: np.asarray(v, dtype=float) for i, v in ct.individual_capacity_limit_kW.items()}

    R = CommunityModel(com.members, "R").build(pooled=False).solve()
    I = CommunityModel(com.members, "I").build(pooled=False, discount=beta, cap_kW=cap_i, penalty=alpha).solve()
    C = CommunityModel(com.members, "C").build(pooled=True, discount=beta, cap_kW=cap, penalty=alpha).solve()

    print("\nValidation of the parametrized model")
    print(f"  R as a limit case (no pooling, no contract): {R.objective:8.2f} DKK")
    print(f"  I from the same model:                       {I.objective:8.2f} DKK")
    print(f"  C <= I:                                      {C.objective:8.2f} <= {I.objective:.2f}  ->  {C.objective <= I.objective + 1e-6}")

    def congestion(res):
        P = res.hourly["P_imp"].to_numpy()
        over = np.maximum(P - cap, 0.0)
        return P.sum(), P.max(), int((over > 1e-6).sum()), over.sum()

    print("\nCongestion at the connection point          R         I         C")
    rows = [congestion(r) for r in (R, I, C)]
    labels = ["daily import (kWh)", "peak import (kW)", "hours above cap", "exceedance (kWh)"]
    for k, lab in enumerate(labels):
        print(f"  {lab:32s}" + "".join(f"{row[k]:10.2f}" for row in rows))
    print(f"  {'daily export (kWh)':32s}" + "".join(f"{r.hourly['P_exp'].sum():10.2f}" for r in (R, I, C)))
    print(f"  {'total cost (DKK)':32s}" + "".join(f"{r.objective:10.2f}" for r in (R, I, C)))

    h = C.hourly.copy()
    h["p+disc_tariff"] = h["price"] + (1 - beta) * com.members[0].import_tariff
    print("\nModel C, hourly: import/export at the meter and internal price lambda (DKK/kWh)")
    print(h[["price", "p+disc_tariff", "P_imp", "P_exp", "E", "lambda", "mu"]].round(3).to_string())
    print(f"\nModel C size: {C.meta['n_vars']} variables, {C.meta['n_constrs']} constraints, "
          f"solved in {C.meta['runtime_s']:.3f} s")

    out_dir.mkdir(parents=True, exist_ok=True)
    C.members.to_csv(out_dir / "C_members.csv", index=False)
    h.to_csv(out_dir / "C_hourly.csv")
    C.q.to_csv(out_dir / "C_net_consumption.csv")

def contract_metrics(com, R, beta: float, f: float) -> dict:
    """Solve I and C for one contract (discount beta, cap tightness f) and return the metrics of Q1.    """
    ct, alpha = com.contract, com.contract.penalty_DKK_per_kWh
    tau_imp, tau_exp = com.members[0].import_tariff, com.members[0].export_tariff
    scale = f / ct.capacity_limit_daily_energy_factor
    cap = scale * np.asarray(ct.capacity_limit_kW, dtype=float)
    cap_i = {i: scale * np.asarray(v, dtype=float) for i, v in ct.individual_capacity_limit_kW.items()}

    I = CommunityModel(com.members, "I").build(pooled=False, discount=beta, cap_kW=cap_i, penalty=alpha).solve()
    C = CommunityModel(com.members, "C").build(pooled=True, discount=beta, cap_kW=cap, penalty=alpha).solve()

    def exceed(res):
        return np.maximum(res.hourly["P_imp"].to_numpy() - cap, 0.0).sum()

    def revenue(res, disc, penalty_kWh):
        """Grid revenue of the DSO: tariffs (discounted under a contract) + penalties, DKK/day."""
        return ((1 - disc) * (tau_imp * res.hourly["P_imp"].sum() + tau_exp * res.hourly["P_exp"].sum())
                + alpha * penalty_kWh)

    E_R, peak_R, rev_R = exceed(R), R.hourly["P_imp"].max(), revenue(R, 0.0, 0.0)
    out = {"beta": beta, "f": f, "cap_kW": cap[0], "cost_R": R.objective, "exceed_R_kWh": E_R}
    for name, res, pen in (("I", I, I.members["exceed_kWh"].sum()), ("C", C, C.hourly["E"].sum())):
        E_X, rev_X = exceed(res), revenue(res, beta, pen)
        out.update({
            f"cost_{name}": res.objective,
            f"value_{name}_DKK": R.objective - res.objective,
            f"value_{name}_pct": 100 * (R.objective - res.objective) / R.objective,
            f"exceed_{name}_kWh": E_X,
            f"exceed_removed_{name}_pct": 100 * (E_R - E_X) / E_R if E_R > 1e-6 else np.nan,
            f"peak_reduction_{name}_pct": 100 * (peak_R - res.hourly["P_imp"].max()) / peak_R,
            f"penalty_{name}_DKK": alpha * pen,
            f"revenue_{name}_DKK": rev_X,
            f"cost_flex_{name}_DKK_per_kWh": (rev_R - rev_X) / (E_R - E_X) if E_R - E_X > 1e-6 else np.nan,
        })
    out["value_pooling_DKK"] = I.objective - C.objective
    out["revenue_R_DKK"] = rev_R
    out["cap_binding_hours_C"] = int((C.hourly["mu"] > 1e-6).sum())
    return out


def run_sensitivity(com, out_dir: Path) -> None:
    """Q1 sensitivity: sweep the discount (at the base cap) and the cap tightness (at the base discount)."""
    betas = [0, 0.25, 0.5, 0.625, 0.75, 0.875, 1]
    fs = [0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 4.0]
    beta0 = com.contract.tariff_discount_ratio
    f0 = com.contract.capacity_limit_daily_energy_factor

    R = CommunityModel(com.members, "R").build(pooled=False).solve()
    sweep_beta = pd.DataFrame([contract_metrics(com, R, b, f0) for b in betas])
    sweep_cap = pd.DataFrame([contract_metrics(com, R, beta0, f) for f in fs])

    # only the metrics the hypotheses test. every metric is still saved in the CSV files
    short = {"value_I_DKK": "value I", "value_C_DKK": "value C", "value_pooling_DKK": "pooling",
             "exceed_removed_C_pct": "exc removed C %", "cost_flex_C_DKK_per_kWh": "cost flex C",
             "cap_binding_hours_C": "binding hours C"}
    print("\nValues in DKK/day against R; cost of flexibility in DKK per kWh removed")
    print(f"\nSweep of the discount (f = {f0}):")
    print(sweep_beta[["beta"] + list(short)].rename(columns=short).round(2).to_string(index=False))
    print(f"\nSweep of the cap tightness (beta = {beta0}):")
    print(sweep_cap[["f", "cap_kW"] + list(short)].rename(columns=short).round(2).to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_beta.to_csv(out_dir / "sweep_discount.csv", index=False)
    sweep_cap.to_csv(out_dir / "sweep_cap.csv", index=False) 

def run_settlement(com, out_dir: Path) -> None:
    """Q1 settlement of model C at the internal prices lambda_t (dual of the community balance).    """
    ct = com.contract
    beta, alpha = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh
    cap = np.asarray(ct.capacity_limit_kW, dtype=float)
    C = CommunityModel(com.members, "C").build(pooled=True, discount=beta, cap_kW=cap, penalty=alpha).solve()
    lam = C.hourly["lambda"].to_numpy()

    rows = []
    for d in com.members:
        i = d.consumer_id
        cost_R = FlexibleConsumerModel(d, name=f"R_{i}").build().solve().objective
        cost_I = FlexibleConsumerModel(d, name=f"I_{i}").build(
            discount=beta, cap_kW=np.asarray(ct.individual_capacity_limit_kW[i], dtype=float),
            penalty=alpha, tariff_over="discounted").solve().objective
        mem = C.members.set_index("consumer_id").loc[i]
        payment = float(lam @ C.q[i].to_numpy())
        cost_C = payment + mem["disutility_DKK"] + mem["pv_cost_DKK"]
        rows.append({"consumer_id": i, "archetype": d.archetype,
                     "net_kWh": C.q[i].sum(), "payment_DKK": payment, "cost_C_DKK": cost_C,
                     "cost_R_DKK": cost_R, "cost_I_DKK": cost_I,
                     "gain_vs_R_DKK": cost_R - cost_C, "gain_vs_I_DKK": cost_I - cost_C})
    s = pd.DataFrame(rows)
    s["IR_vs_R"] = s["gain_vs_R_DKK"] >= -1e-6
    s["IR_vs_I"] = s["gain_vs_I_DKK"] >= -1e-6

    collected = s["payment_DKK"].sum()
    bill = C.meta["bill_DKK"]
    rent = float((C.hourly["mu"] * cap).sum())

    print("\nSettlement of C at the internal prices (DKK/day; gain > 0: the member is better off in C)")
    print(s.round(2).to_string(index=False))
    print("\nBudget of the community manager (DKK/day)")
    print(f"  collected from the members   {collected:8.2f}")
    print(f"  bill paid to the DSO         {bill:8.2f}")
    print(f"  surplus                      {collected - bill:8.2f}   (congestion rent sum_t mu_t x cap = {rent:.2f})")
    print("\nIndividual rationality")
    print(f"  members better off than in R: {int(s.IR_vs_R.sum())} of {len(s)}")
    print(f"  members better off than in I: {int(s.IR_vs_I.sum())} of {len(s)}")
    print(f"  total gain vs R: {s.gain_vs_R_DKK.sum():.2f}  (= value of C {s.cost_R_DKK.sum() - C.objective:.2f}"
          f" minus surplus {collected - bill:.2f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_dir / "settlement_members.csv", index=False)

def run_q2_sensitivity(com, out_dir: Path, time_limit: float = 120) -> None:
    """Q2 sensitivity on the discount beta: DP with the cap only and with budget balance, at every beta.

    Reports per step: status (feasibility), efficiency loss against C at the same beta (DKK/day), the
    manager's surplus (DKK/day), and the members worse off than in R, i.e. the members blocking
    individual rationality, with their total loss (DKK/day). Writes q2_sensitivity_beta.csv.
    """
    ct = com.contract
    betas = [0, 0.25, 0.5, 0.75, 1]
    R = {d.consumer_id: FlexibleConsumerModel(d).build().solve().objective for d in com.members}

    rows = []
    for beta in betas:
        C = CommunityModel(com.members, "C").build(pooled=True, discount=beta, cap_kW=ct.capacity_limit_kW,
                                                   penalty=ct.penalty_DKK_per_kWh).solve()
        for variant in ("cap", "bb"):
            r = solve_dp(build_dp(com, variant, R, discount=beta, x_ub=None if variant == "cap" else np.inf), time_limit=time_limit)
            row = {"beta": beta, "variant": variant, "status": r["status"], "C_cost_DKK": C.objective,
                   "runtime_s": r["runtime_s"]}
            if r["has_solution"]:
                cost = r["members"].set_index("consumer_id").cost_DKK
                loss = (cost - pd.Series(R)).clip(lower=0)
                blocking = loss[loss > 1e-5]
                row.update({"DP_cost_DKK": r["objective_DKK"],
                            "efficiency_loss_DKK": r["objective_DKK"] - C.objective,
                            "surplus_DKK": r["budget_surplus_DKK"],
                            "n_worse_off_than_R": len(blocking),
                            "total_IR_violation_DKK": blocking.sum(),
                            "blocking_members": ",".join(blocking.index)})
            rows.append(row)
            print(f"beta={beta:<5} {variant:<4} {r['status']}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nQ2 sensitivity on the discount (DKK/day; efficiency loss against C at the same beta)")
    print(df[["beta", "variant", "status", "efficiency_loss_DKK", "surplus_DKK", "n_worse_off_than_R",
              "total_IR_violation_DKK", "blocking_members"]].round(2).to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "q2_sensitivity_beta.csv", index=False)

def run_q2(com, out_dir: Path, time_limit: float = 120) -> None:
    """Q2 base case: DP in three steps, cap only -> + budget balance -> + individual rationality.

    Individual rationality (cost <= cost in R for every member) is tested through the IR relaxation:
    budget balance is kept and the total IR violation is minimized, warm-started from the budget-balanced
    solution. A total violation of 0 means that uniform prices meet both requirements.
    Prints the comparison with C and the KKT residuals; writes DP_<variant>_hourly.csv,
    DP_<variant>_members.csv and q2_overview.csv.
    """
    ct = com.contract
    R = {d.consumer_id: FlexibleConsumerModel(d).build().solve().objective for d in com.members}
    C = CommunityModel(com.members, "C").build(pooled=True, discount=ct.tariff_discount_ratio,
                                               cap_kW=ct.capacity_limit_kW, penalty=ct.penalty_DKK_per_kWh).solve()

    results, start = {}, None
    for variant in ("cap", "bb"):
        b = build_dp(com, variant, R, x_ub=None if variant == "cap" else np.inf)
        results[variant] = r = solve_dp(b, time_limit=time_limit)
        print(f"DP {variant:<12} {r['status']}")
        if variant == "bb" and r["has_solution"]:
            start = solution_values(b)
            cost = r["members"].set_index("consumer_id").cost_DKK
            start.update({f"IR_relax[{i}]": max(0.0, cost[i] - R[i]) for i in R})
    results["bb_ir_relax"] = r = solve_dp(build_dp(com, "bb_ir_relax", R, min_relax=True),
                                          time_limit=time_limit, start=start)
    print(f"DP bb_ir_relax  {r['status']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, r in results.items():
        row = {"variant": variant, "status": r["status"], "runtime_s": r["runtime_s"],
               "best_bound": r.get("best_bound")}
        if r["has_solution"]:
            if max(r["max_payment_gap"], r["max_stationarity"], r["max_complementarity"]) > 1e-4:
                raise RuntimeError(f"KKT validation failed for DP {variant}")
            members = r["members"]
            members["cost_R_DKK"] = members.consumer_id.map(R)
            members["gain_vs_R_DKK"] = members.cost_R_DKK - members.cost_DKK
            losers = members.loc[members.gain_vs_R_DKK < -1e-5]
            row.update({"welfare_DKK": r["objective_DKK"], "efficiency_loss_DKK": r["objective_DKK"] - C.objective,
                        "surplus_DKK": r["budget_surplus_DKK"], "n_worse_off_than_R": len(losers),
                        "total_IR_violation_DKK": -losers.gain_vs_R_DKK.sum(),
                        "blocking_members": ",".join(losers.consumer_id),
                        "max_KKT_residual": max(r["max_payment_gap"], r["max_stationarity"],
                                                r["max_complementarity"])})
            r["hourly"].to_csv(out_dir / f"DP_{variant}_hourly.csv", index=False)
            members.to_csv(out_dir / f"DP_{variant}_members.csv", index=False)
        rows.append(row)

    overview = pd.DataFrame(rows)
    overview.to_csv(out_dir / "q2_overview.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\nQ2 base case (DKK/day). C: {C.objective:.2f}; members' total cost in R: {sum(R.values()):.2f}")
    print(overview.drop(columns=["blocking_members"], errors="ignore").round(4).to_string(index=False))
    for row in rows:
        if row.get("blocking_members"):
            print(f"{row['variant']}: worse off than R -> {row['blocking_members']}")

def run_q3_baseline(com, out_dir: Path, n: int = 100, seed: int = 100) -> None:
    """Q3 step 1: the deterministic baseline DP (budget balance, theta = 1) on n out-of-sample scenarios.

    Draws the scenarios (seed reported; the theta of every member and scenario is saved), posts the budget-balanced
    DP prices, and evaluates them with optimistic tie-breaking. Prints the distribution over the scenarios of the
    welfare cost, penalty, cap exceedance, manager's surplus and IR violation, and traces the worst scenarios back
    to their theta. Writes q3_scenarios_seed<seed>_theta.csv and q3_baseline_<scenario|member>.csv.
    """
    R = {d.consumer_id: FlexibleConsumerModel(d).build().solve().objective for d in com.members}
    dp = solve_dp(build_dp(com, "bb", R), time_limit=120)
    x = dp["hourly"].x.to_numpy()

    # theta = 1 for every member: the optimistic response must reproduce DP; the solver's arbitrary
    base = draw_scenarios(com, n=1, seed=seed)
    print(f"\nDP (budget balance): welfare cost {dp['objective_DKK']:.2f}, surplus {dp['budget_surplus_DKK']:.2f}")
    print("Same prices with theta = 1 for every member:")
    for tb in ("optimistic", "solver"):
        r = evaluate_prices(com, x, base, tie_break=tb)[0].iloc[0]
        print(f"  {tb:<10} welfare cost {r.welfare_cost_DKK:8.2f} | exceedance {r.exceedance_kWh:6.2f} kWh"
              f" | surplus {r.surplus_DKK:8.2f} | members worse off than R {r.n_worse_off_than_R}")

    S = draw_scenarios(com, n=n, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"scenario": s["id"], **s["theta"]["profile"]} for s in S]).to_csv(
        out_dir / f"q3_scenarios_seed{seed}_theta.csv", index=False)
    per_s, per_m = evaluate_prices(com, x, S)
    per_s.to_csv(out_dir / "q3_baseline_scenario.csv", index=False)
    per_m.to_csv(out_dir / "q3_baseline_member.csv", index=False)

    cols = ["welfare_cost_DKK", "penalty_DKK", "exceedance_kWh", "hours_above_cap", "surplus_DKK",
            "n_worse_off_than_R", "total_IR_violation_DKK"]
    summary = per_s[cols].describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "min", "5%", "50%", "95%", "max"]]
    print(f"\nQ3 baseline: budget-balanced DP prices on {n} out-of-sample scenarios (seed {seed})")
    print(summary.round(2).to_string())
    print(f"\nscenarios with cap exceedance: {int((per_s.exceedance_kWh > 1e-6).sum())} of {n}"
          f" | with a budget deficit (surplus < 0): {int((per_s.surplus_DKK < -1e-6).sum())} of {n}")

    print("\nWorst scenarios (highest welfare cost) and the members with the lowest theta (wanting more energy):")
    for _, w in per_s.nlargest(3, "welfare_cost_DKK").iterrows():
        th = per_m[per_m.scenario == w.scenario].nsmallest(4, "theta")
        print(f"  scenario {int(w.scenario)}: cost {w.welfare_cost_DKK:.2f}, exceedance {w.exceedance_kWh:.2f} kWh, "
              f"surplus {w.surplus_DKK:.2f} | " + ", ".join(f"{r.consumer_id} {r.theta:.2f}" for r in th.itertuples()))     

def _q3_summary(per_s: pd.DataFrame, label: str) -> dict:
    """Metrics of one set of evaluated scenarios: the cost as a distribution (mean, 95% quantile, CVaR95 = mean of
    the worst 5%), cap and budget satisfaction (frequency and magnitude), and individual rationality."""
    c = per_s.welfare_cost_DKK
    worst = c.nlargest(max(1, int(np.ceil(0.05 * len(c)))))
    return {"case": label, "scenarios": len(per_s),
            "mean_cost_DKK": c.mean(), "q95_cost_DKK": c.quantile(0.95), "CVaR95_cost_DKK": worst.mean(),
            "P_exceed_%": 100 * (per_s.exceedance_kWh > 1e-6).mean(),
            "mean_exceed_kWh": per_s.exceedance_kWh.mean(), "max_exceed_kWh": per_s.exceedance_kWh.max(),
            "P_deficit_%": 100 * (per_s.surplus_DKK < -1e-4).mean(),
            "mean_surplus_DKK": per_s.surplus_DKK.mean(), "worst_deficit_DKK": max(0.0, -per_s.surplus_DKK.min()),
            "mean_worse_off": per_s.n_worse_off_than_R.mean(),
            "mean_IR_violation_DKK": per_s.total_IR_violation_DKK.mean()}
    

def _sdp_solve_or_load(com, S_in, label: str, params: dict, tag: str, out_dir: Path, x_start,
                       time_limit: float, resolve: bool):
    """Solve one SDP variant on the in-sample scenarios and save its prices and solve details, or reload them."""
    f_x, f_info = out_dir / f"{tag}_prices.csv", out_dir / f"{tag}_solve.csv"
    if f_x.exists() and f_info.exists() and not resolve:
        print(f"Loaded the {label} prices from {f_x.name} (add --q3-resolve to solve again)")
        return pd.read_csv(f_x).x.to_numpy(), {**pd.read_csv(f_info).iloc[0].to_dict(), "model": label}
    if f_x.exists():                         # resolving: start from this model's previous solution
        x_start = pd.read_csv(f_x).x.to_numpy()
    print(f"Solving {label} ({tag}, time limit {time_limit:.0f} s) ...")
    r = StochasticDP(com, S_in).build(**params).solve(time_limit=time_limit, x_start=x_start)
    if not r["has_solution"]:
        print(f"  {label}: no solution within {time_limit:.0f} s ({r['status']}); skipped")
        return None, {"model": label, "status": r["status"], "N": r["N"], "runtime_s": r["runtime_s"],
                      "n_vars": r["n_vars"], "n_sos": r["n_sos"]}
    if r["max_KKT_residual"] > 1e-4:
        raise RuntimeError(f"{tag}: KKT validation failed")
    keys = ("status", "N", "runtime_s", "mip_gap", "best_bound", "objective_DKK", "expected_cost_DKK",
            "max_KKT_residual", "n_vars", "n_constrs", "n_sos")
    info = {"model": label, **{k: r[k] for k in keys}}
    pd.DataFrame([info]).to_csv(f_info, index=False)
    pd.DataFrame({"hour": range(len(r["x"])), "x": r["x"]}).to_csv(f_x, index=False)
    return r["x"], info

    
def run_q3_validate(com, n: int = 2, seed: int = 1, time_limit: float = 120) -> None:
    """Q3 validation of the variants on a small scenario set: the limit cases beta_r = 0 (CVaR) and eps = 1 (chance
    constraint) must reproduce SDP^b, and the predicted ordering of the in-sample expected costs must hold."""
    S = draw_scenarios(com, n=n, seed=seed)
    runs = {"SDP^b": {},
            "SDP^CVaR beta_r=0": {"objective": "cvar", "alpha": 1.0 / n, "beta_r": 0.0},
            "SDP^CC eps=1": {"budget": "chance", "eps": 1.0},
            "SDP^CVaR beta_r=0.5": {"objective": "cvar", "alpha": 1.0 / n, "beta_r": 0.5},
            "SDP^CC eps=0": {"budget": "chance", "eps": 0.0}}
    rows = []
    for label, params in runs.items():
        r = StochasticDP(com, S).build(**params).solve(time_limit=time_limit)
        worst = r["scenarios"].cost_DKK.max() if r["has_solution"] else np.nan
        rows.append({"model": label, "status": r["status"], "gap_%": 100 * r["mip_gap"],
                     "expected_cost_DKK": r.get("expected_cost_DKK", np.nan), "worst_scenario_DKK": worst,
                     "max_KKT_residual": r.get("max_KKT_residual", np.nan), "runtime_s": r["runtime_s"]})
        print(f"{label:<22} {r['status']}")
    df = pd.DataFrame(rows).set_index("model")
    print(f"\nValidation of the variants (N = {n}, seed {seed})")
    print(df.round(4).to_string())
    base = df.loc["SDP^b", "expected_cost_DKK"]
    tol = 1e-3 * abs(base)
    print("\nLimit cases reproduce SDP^b:",
        all(abs(df.loc[k, "expected_cost_DKK"] - base) <= tol for k in ("SDP^CVaR beta_r=0", "SDP^CC eps=1")))
    print("Predicted ordering (variants >= SDP^b in expected cost):",
        all(df.loc[k, "expected_cost_DKK"] >= base - tol for k in ("SDP^CVaR beta_r=0.5", "SDP^CC eps=0")))
    
def run_q3_sdp(com, out_dir: Path, models=("b", "cvar", "cc"), n: int = 5, seed: int = 1, n_out: int = 100,
               seed_out: int = 100, time_limit: float = 300, resolve: bool = False,
               alpha: float = 0.2, beta_r: float = 0.5, eps: float = 0.2, deficit_tol: float = 0.0) -> None:
    """Q3: the stochastic models against the deterministic baseline DP, in-sample and out-of-sample.

    models: any of "b" (SDP^b: expected cost, budget balance in expectation), "cvar" (SDP^CVaR: weight beta_r on
    the CVaR of the worst alpha fraction of scenarios) and "cc" (SDP^CC: no deficit with probability >= 1 - eps).
    Each model is solved once on the n in-sample scenarios (seed) and its prices are saved; later runs reload them
    unless resolve=True. All price vectors are evaluated on the in-sample scenarios and on the same n_out
    out-of-sample scenarios (seed_out). Prints and writes the solve details and the comparison table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = {"b": ("SDP^b", "SDP_b", {}),
               "cvar": ("SDP^CVaR", f"SDP_CVaR_a{alpha:g}_b{beta_r:g}",
                        {"objective": "cvar", "alpha": alpha, "beta_r": beta_r}),
               "cc": ("SDP^CC", f"SDP_CC_eps{eps:g}_tol{deficit_tol:g}",
                      {"budget": "chance", "eps": eps, "deficit_tol": deficit_tol})}
    R = {d.consumer_id: FlexibleConsumerModel(d).build().solve().objective for d in com.members}
    x_dp = solve_dp(build_dp(com, "bb", R), time_limit=120)["hourly"].x.to_numpy()
    S_in = draw_scenarios(com, n=n, seed=seed)
    S_out = draw_scenarios(com, n=n_out, seed=seed_out)
    pd.DataFrame([{"scenario": s["id"], **s["theta"]["profile"]} for s in S_in]).to_csv(
        out_dir / f"q3_scenarios_seed{seed}_N{n}_theta.csv", index=False)

    prices, infos = {"DP": x_dp}, []
    for key in models:
        label, name, params = configs[key]
        tag = f"{name}_N{n}_seed{seed}"
        x_start = prices.get("SDP^b", x_dp)          # the variants start from the SDP^b prices
        x, info = _sdp_solve_or_load(com, S_in, label, params, tag, out_dir, x_start, time_limit, resolve)
        infos.append(info)
        if x is None:                        # no solution found: reported in the solve details, not evaluated
            continue
        prices[label] = x
        diff = np.flatnonzero(np.abs(x - x_dp) > 1e-3)
        print(f"  {label} vs DP prices: " + (", ".join(f"h{t} {x_dp[t]:.2f}->{x[t]:.2f}" for t in diff) or "same"))

    pd.set_option("display.width", 250)
    print(f"\nSolve details (N = {n}, seed {seed})")
    print(pd.DataFrame(infos).round(4).to_string(index=False))

    rows = []
    for label, x in prices.items():
        for set_name, S in (("in", S_in), ("out", S_out)):
            print(f"Evaluating {label} {set_name}-sample ({len(S)} scenarios) ...")
            per_s, _ = evaluate_prices(com, x, S)
            per_s.to_csv(out_dir / f"q3_eval_{label.replace('^', '_')}_N{n}_seed{seed}_{set_name}.csv", index=False)
            rows.append(_q3_summary(per_s, f"{label} {set_name}"))
    table = pd.DataFrame(rows).set_index("case").T
    table.to_csv(out_dir / f"q3_compare_N{n}_seed{seed}.csv")
    print(f"\nModels in-sample (N = {n}, seed {seed}) and out-of-sample ({n_out} scenarios, seed {seed_out})")
    print(table.round(2).to_string())
        
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--reference", action="store_true", help="Q1: solve model R and check the congestion")
    ap.add_argument("--individual", action="store_true", help="Q1: solve model I and compare it with R")
    ap.add_argument("--central", action="store_true", help="Q1: solve model C (community contract) and compare with R and I")
    ap.add_argument("--sensitivity", action="store_true", help="Q1: sweep the discount and the cap tightness")
    ap.add_argument("--settlement", action="store_true", help="Q1: settle C at the internal prices, test budget balance and IR")
    ap.add_argument("--q2-sensitivity", action="store_true", help="Q2: DP (cap, budget balance) over the discount; blocking members")
    ap.add_argument("--q2", action="store_true", help="Q2: DP on the base case, cap -> budget balance -> IR")
    ap.add_argument("--q3-baseline", action="store_true", help="Q3: deterministic DP prices on 100 out-of-sample scenarios")    
    ap.add_argument("--q3-sdp", action="store_true", help="Q3: SDP^E vs DP, in-sample and out-of-sample")
    ap.add_argument("--q3-models", default="b,cvar,cc", help="Q3: models to solve, comma-separated: b, cvar, cc")
    ap.add_argument("--q3-alpha", type=float, default=0.2, help="Q3: CVaR tail fraction (a multiple of 1/N)")
    ap.add_argument("--q3-beta-r", type=float, default=0.5, help="Q3: CVaR weight in the objective")
    ap.add_argument("--q3-eps", type=float, default=0.2, help="Q3: accepted deficit probability in SDP^CC")
    ap.add_argument("--q3-validate", action="store_true", help="Q3: validate the variants on 2 scenarios")
    ap.add_argument("--q3-n", type=int, default=5, help="Q3: number of in-sample scenarios N")
    ap.add_argument("--q3-seed", type=int, default=1, help="Q3: seed of the in-sample scenarios")
    ap.add_argument("--q3-time-limit", type=float, default=300, help="Q3: time limit of each SDP solve (s)")
    ap.add_argument("--q3-resolve", action="store_true", help="Q3: solve the SDP models again even if their prices are saved")
    ap.add_argument("--q3-deficit-tol", type=float, default=0.0, help="Q3: deficit tolerated in SDP^CC (DKK/day)")

        
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")
    if args.reference:
        run_reference(com, RESULTS_DIR / args.community / args.day)
    if args.individual:
        run_individual(com, RESULTS_DIR / args.community / args.day)
    if args.central:
        run_central(com, RESULTS_DIR / args.community / args.day)
    if args.sensitivity:
        run_sensitivity(com, RESULTS_DIR / args.community / args.day)
    if args.settlement:
        run_settlement(com, RESULTS_DIR / args.community / args.day)
    if args.q2_sensitivity:
        run_q2_sensitivity(com, RESULTS_DIR / args.community / args.day / "Q2")
    if args.q2:
        run_q2(com, RESULTS_DIR / args.community / args.day / "Q2")
    if args.q3_baseline:
        run_q3_baseline(com, RESULTS_DIR / args.community / args.day / "Q3")
    if args.q3_sdp:
        run_q3_sdp(com, RESULTS_DIR / args.community / args.day / "Q3", models=args.q3_models.split(","),
                   n=args.q3_n, seed=args.q3_seed, time_limit=args.q3_time_limit, resolve=args.q3_resolve,
                   alpha=args.q3_alpha, beta_r=args.q3_beta_r, eps=args.q3_eps, deficit_tol=args.q3_deficit_tol)

    if args.q3_validate:
        run_q3_validate(com)


if __name__ == "__main__":
    main()
