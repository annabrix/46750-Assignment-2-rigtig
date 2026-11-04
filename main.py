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



if __name__ == "__main__":
    main()
