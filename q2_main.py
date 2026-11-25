
"""Run Question 2 experiments.

Examples:
    python main.py --q2-best-response
    python main.py --q2 cap
    python main.py --q2 all
    python main.py --q2-sensitivity
"""

import argparse
from pathlib import Path


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from src.data_loader import load_community
from src.model import FlexibleConsumerModel
from src.q2_dynamic_pricing import (
    solve_central,
    best_response,
    build_dp,
    solve_dp,
    indifference,
)

RESULTS = Path(__file__).resolve().parent / "results"

BETAS = [0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]


# ============================================================
# QUESTION 1 BENCHMARKS NEEDED FOR Q2
# ============================================================

def benchmarks(com):
    """Compute individual reference R and contract I costs."""

    R, I = {}, {}
    ct = com.contract

    for d in com.members:

        R[d.consumer_id] = (
            FlexibleConsumerModel(d)
            .build()
            .solve()
            .objective
        )

        I[d.consumer_id] = (
            FlexibleConsumerModel(d)
            .build(
                discount=ct.tariff_discount_ratio,
                cap_kW=ct.individual_capacity_limit_kW[
                    d.consumer_id
                ],
                penalty=ct.penalty_DKK_per_kWh,
                tariff_over="discounted"
            )
            .solve()
            .objective
        )

    return R, I


# ============================================================
# FIGURE HELPERS
# ============================================================

def save_figure(folder, name):

    folder.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()

    plt.savefig(
        folder / (name + ".pdf"),
        bbox_inches="tight"
    )

    plt.savefig(
        folder / (name + ".png"),
        dpi=170,
        bbox_inches="tight"
    )

    plt.close()


def plot_load(com, lam, figs):
    """Figure 1: flexible vs less flexible member."""

    plt.figure(figsize=(9, 3.7))

    for cid in ("C1", "C4"):

        d = next(
            d for d in com.members
            if d.consumer_id == cid
        )

        r = best_response(d, lam)

        plt.plot(
            range(len(lam)),
            r["load"],
            marker="o",
            markersize=2.5,
            label=f"{cid}, cL={d.linear_disutility}"
        )

    plt.plot(
        range(len(lam)),
        d.reference_load,
        linestyle="--",
        linewidth=1.5,
        label="Reference load (both members)"
    )

    plt.xlabel("Hour")
    plt.ylabel("Load (kWh)")

    plt.title(
        "Step 1: individual responses "
        "to centralized dual prices"
    )

    plt.legend()

    save_figure(figs, "fig_q2_load_response")


def plot_prices(C, results, figs):
    """Figure 2: internal prices from C and DP variants."""

    plt.figure(figsize=(9, 3.7))

    plt.plot(
        C["hourly"]["hour"],
        C["hourly"]["lambda"],
        label="Centralized dual price (C)",
        linewidth=2.2
    )

    for variant in ("cap", "bb", "bb_ir"):

        r = results.get(variant, {})

        if r.get("status") == "OPTIMAL":

            plt.plot(
                r["hourly"]["hour"],
                r["hourly"]["x"],
                marker=".",
                label=f"DP {variant}"
            )

    plt.xlabel("Hour")
    plt.ylabel("Internal price (DKK/kWh)")

    plt.title(
        "Step 3: posted prices versus "
        "centralized marginal prices"
    )

    plt.legend()

    save_figure(figs, "fig_q2_prices")


def plot_gains(com, results, figs):
    """Figure 3: individual gains relative to R."""

    variants = [
        v for v in ("cap", "bb", "bb_ir")
        if results.get(v, {}).get("status") == "OPTIMAL"
    ]

    if not variants:
        return

    names = [
        d.consumer_id for d in com.members
    ]

    values = np.array([
        [
            results[v]["members"]
            .set_index("consumer_id")
            .loc[i, "gain_vs_R_DKK"]

            for v in variants
        ]
        for i in names
    ])

    fig, ax = plt.subplots(figsize=(7, 7))

    scale = max(abs(values).max(), 0.001)

    heat = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu",
        vmin=-scale,
        vmax=scale
    )

    ax.set_xticks(range(len(variants)), variants)
    ax.set_yticks(range(len(names)), names)

    ax.set_title(
        "Step 4: individual gain versus R (DKK)"
    )

    fig.colorbar(
        heat,
        ax=ax,
        label="Gain (DKK); negative = IR violation"
    )

    for i in range(len(names)):
        for j in range(len(variants)):
            ax.text(
                j, i,
                f"{values[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8
            )

    save_figure(figs, "fig_q2_member_gains")


def plot_sensitivity(sweep, figs):
    """Figure 4: efficiency loss vs tariff discount."""

    plt.figure(figsize=(9, 3.8))

    for variant in ("cap", "bb", "bb_ir"):

        z = sweep[
            (sweep.variant == variant)
            & (sweep.status == "OPTIMAL")
        ]

        if not z.empty:

            plt.plot(
                z.beta,
                z.efficiency_loss_DKK,
                marker="o",
                label=variant
            )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("DSO tariff discount beta")
    plt.ylabel("Cost above central C (DKK/day)")

    plt.title(
        "Step 5: efficiency loss versus contract discount"
    )

    plt.legend()

    save_figure(figs, "fig_q2_sensitivity")


# ============================================================
# STEP 1: VALIDATE MEMBERS' BEST RESPONSES
# ============================================================

def run_best_response(com, output):

    output.mkdir(parents=True, exist_ok=True)
    figs = output / "figures"

    C = solve_central(com)

    lam = C["hourly"]["lambda"].to_numpy()

    rows = []

    for d in com.members:

        i = d.consumer_id

        r = best_response(d, lam)

        # Cost of original C dispatch at internal prices
        original = float(
            lam @ C["q"][i] + C["D"][i]
        )

        rows.append({
            "consumer_id": i,
            "cost_at_C_dispatch": original,
            "best_response_cost": r["cost"],
            "optimality_gap": original - r["cost"],
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        output / "q2_best_response_validation.csv",
        index=False
    )

    plot_load(com, lam, figs)

    print(df.round(7).to_string(index=False))

    print(
        "Maximum |member optimality gap|:",
        df.optimality_gap.abs().max()
    )

    print("Saved fig_q2_load_response.pdf")


# ============================================================
# STEP 4: INDIVIDUAL BENEFITS
# ============================================================

def add_benefits(r, R, I):

    if not r["has_solution"]:
        return

    members = r["members"]

    members["reference_R_DKK"] = (
        members.consumer_id.map(R)
    )

    members["reference_I_DKK"] = (
        members.consumer_id.map(I)
    )

    members["gain_vs_R_DKK"] = (
        members.reference_R_DKK - members.cost_DKK
    )

    members["gain_vs_I_DKK"] = (
        members.reference_I_DKK - members.cost_DKK
    )

    r["n_IR_violations"] = int(
        (members.gain_vs_R_DKK < -1e-5).sum()
    )


def save_result(r, output):

    if r.get("has_solution"):

        r["hourly"].to_csv(
            output / f"DP_{r['variant']}_hourly.csv",
            index=False
        )

        r["members"].to_csv(
            output / f"DP_{r['variant']}_members.csv",
            index=False
        )


# ============================================================
# STEP 4: MINIMUM IR RELAXATION IF BB+IR IS INFEASIBLE
# ============================================================

def ir_relaxation(
    com,
    R,
    *,
    beta=None,
    limit=300,
    price_bound=None,
    verbose=False
):
    """Minimize total IR violations, then welfare."""

    b = build_dp(
        com,
        "bb_ir_relax",
        R,
        discount=beta,
        price_bound=price_bound,
        min_relax=True,
        verbose=verbose
    )

    first = solve_dp(
        b,
        time_limit=limit,
        verbose=verbose
    )

    if first["status"] != "OPTIMAL":
        return first

    minimum = first["solver_objective"]

    m = b["model"]

    m.addConstr(
        gp.quicksum(b["relax"].values())
        <= minimum + 1e-6,
        name="fix_minimum_IR_relaxation"
    )

    m.setObjective(
        b["welfare"],
        GRB.MINIMIZE
    )

    second = solve_dp(
        b,
        time_limit=limit,
        verbose=verbose
    )

    second["minimum_IR_relax_DKK"] = minimum

    return second


# ============================================================
# STEPS 2-4: RUN DYNAMIC PRICING VARIANTS
# ============================================================

def run_variants(
    com,
    output,
    requested,
    limit,
    price_bound,
    verbose,
    check_indifference
):

    output.mkdir(parents=True, exist_ok=True)
    figs = output / "figures"

    R, I = benchmarks(com)
    C = solve_central(com)

    print(
        f"Central C benchmark: "
        f"{C['cost']:.4f} DKK/day"
    )

    results = {}

    options = (
        ("cap", "bb", "bb_ir")
        if requested == "all"
        else (requested,)
    )

    for variant in options:

        print(f"\nSolving DP {variant} ...")

        b = build_dp(
            com,
            variant,
            R,
            price_bound=price_bound,
            verbose=verbose
        )

        r = solve_dp(
            b,
            time_limit=limit,
            verbose=verbose
        )

        add_benefits(r, R, I)

        results[variant] = r

        print(
            "Status:", r["status"],
            "  MIP gap:", r["mip_gap"]
        )

        if r["has_solution"]:

            print(
                f"Welfare: {r['objective_DKK']:.3f}"
                f" | C: {C['cost']:.3f}"
                f" | Efficiency loss: "
                f"{r['objective_DKK']-C['cost']:.3f}"
            )

            print(
                f"Budget surplus: "
                f"{r['budget_surplus_DKK']:.4f}"
                f" | IR violations: "
                f"{r['n_IR_violations']}"
            )

            print(
                f"KKT residuals: "
                f"payment {r['max_payment_gap']:.2e}, "
                f"stationarity "
                f"{r['max_stationarity']:.2e}, "
                f"complementarity "
                f"{r['max_complementarity']:.2e}"
            )

            if (
                r["max_payment_gap"] > 1e-3
                or r["max_complementarity"] > 1e-3
            ):
                raise RuntimeError(
                    "KKT/payment validation failed"
                )

        save_result(r, output)

    # If combined requirements are infeasible:
    if (
        results.get("bb_ir", {}).get("status")
        == "INFEASIBLE"
    ):

        print(
            "\nBB+IR infeasible. "
            "Solving minimum-IR-relaxation problem ..."
        )

        rr = ir_relaxation(
            com,
            R,
            limit=limit,
            price_bound=price_bound,
            verbose=verbose
        )

        results["bb_ir_relax"] = rr

        add_benefits(rr, R, I)
        save_result(rr, output)

        if rr.get("has_solution"):

            print(
                rr["members"].loc[
                    lambda x: x.IR_relax_DKK > 1e-5,
                    ["consumer_id", "IR_relax_DKK"]
                ].to_string(index=False)
            )

    # Save summary table
    rows = []

    for v, r in results.items():

        rows.append({
            "variant": v,
            "status": r["status"],
            "welfare_DKK":
                r.get("objective_DKK", np.nan),
            "C_cost_DKK": C["cost"],
            "efficiency_loss_DKK":
                r.get("objective_DKK", np.nan)
                - C["cost"],
            "budget_surplus_DKK":
                r.get("budget_surplus_DKK", np.nan),
            "cap_exceedance_kWh":
                r.get("cap_exceedance_kWh", np.nan),
            "IR_violations_R":
                r.get("n_IR_violations", np.nan),
            "min_IR_relax_DKK":
                r.get("minimum_IR_relax_DKK", np.nan),
            "mip_gap": r["mip_gap"],
            "runtime_s": r["runtime_s"],
        })

    pd.DataFrame(rows).to_csv(
        output / "q2_overview.csv",
        index=False
    )

    # Generate figures for the report
    plot_load(
        com,
        C["hourly"]["lambda"].to_numpy(),
        figs
    )

    plot_prices(C, results, figs)
    plot_gains(com, results, figs)

    # Optional indifference analysis
    if check_indifference:

        if (
            results.get("cap", {}).get("status")
            == "OPTIMAL"
        ):

            check = indifference(
                com,
                results["cap"]["hourly"]["x"].to_numpy()
            )

            check.to_csv(
                output / "q2_indifference_hours.csv",
                index=False
            )

            print(
                f"Indifferent member-hours detected: "
                f"{len(check)}"
            )

        else:
            print(
                "Indifference requires a solved "
                "optimal 'cap' variant"
            )

    print("Results saved in:", output)


# ============================================================
# STEP 5: TARIFF DISCOUNT SENSITIVITY
# ============================================================

def run_sensitivity(
    com,
    output,
    limit,
    price_bound,
    verbose
):

    output.mkdir(parents=True, exist_ok=True)
    figs = output / "figures"

    R, _ = benchmarks(com)
    records = []

    for beta in BETAS:

        C = solve_central(
            com,
            discount=beta
        )

        for variant in ("cap", "bb", "bb_ir"):

            print(
                f"Sensitivity beta={beta}, "
                f"variant={variant}"
            )

            b = build_dp(
                com,
                variant,
                R,
                discount=beta,
                price_bound=price_bound,
                verbose=verbose
            )

            r = solve_dp(
                b,
                time_limit=limit,
                verbose=verbose
            )

            blocking = ""

            if r["has_solution"]:

                members = r["members"]

                gains = (
                    members.consumer_id.map(R)
                    - members.cost_DKK
                )

                blocking = ",".join(
                    members.loc[
                        gains < -1e-5,
                        "consumer_id"
                    ]
                )

            relax_val = np.nan

            if (
                variant == "bb_ir"
                and r["status"] == "INFEASIBLE"
            ):

                rr = ir_relaxation(
                    com,
                    R,
                    beta=beta,
                    limit=limit,
                    price_bound=price_bound,
                    verbose=verbose
                )

                relax_val = rr.get(
                    "minimum_IR_relax_DKK",
                    np.nan
                )

                if rr.get("has_solution"):

                    blocking = ",".join(
                        rr["members"].loc[
                            lambda d:
                                d.IR_relax_DKK > 1e-5,
                            "consumer_id"
                        ]
                    )

            records.append({
                "beta": beta,
                "variant": variant,
                "status": r["status"],
                "C_cost_DKK": C["cost"],
                "DP_cost_DKK":
                    r.get("objective_DKK", np.nan),
                "efficiency_loss_DKK":
                    r.get("objective_DKK", np.nan)
                    - C["cost"],
                "budget_surplus_DKK":
                    r.get("budget_surplus_DKK", np.nan),
                "cap_exceedance_kWh":
                    r.get("cap_exceedance_kWh", np.nan),
                "blocking_members": blocking,
                "min_IR_relax_DKK": relax_val,
                "mip_gap": r["mip_gap"],
                "runtime_s": r["runtime_s"],
            })

    sweep = pd.DataFrame(records)

    sweep.to_csv(
        output / "q2_sensitivity_beta.csv",
        index=False
    )

    plot_sensitivity(sweep, figs)

    print(sweep.to_string(index=False))
    print("Saved:", output)


# ============================================================
# MAIN: COMMAND-LINE OPTIONS
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Question 2: dynamic pricing"
    )

    parser.add_argument(
        "--community",
        default="base"
    )

    parser.add_argument(
        "--day",
        default="d1"
    )

    parser.add_argument(
        "--q2-best-response",
        action="store_true"
    )

    parser.add_argument(
        "--q2",
        choices=["cap", "bb", "bb_ir", "all"]
    )

    parser.add_argument(
        "--q2-indifference",
        action="store_true"
    )

    parser.add_argument(
        "--q2-sensitivity",
        action="store_true"
    )

    parser.add_argument(
        "--q2-time-limit",
        type=float,
        default=300
    )

    parser.add_argument(
        "--q2-price-bound",
        type=float,
        default=None
    )

    parser.add_argument(
        "--q2-verbose",
        action="store_true"
    )

    args = parser.parse_args()

    com = load_community(
        args.community,
        day=args.day
    )

    output = (
        RESULTS
        / args.community
        / args.day
        / "Q2"
    )

    if args.q2_best_response:
        run_best_response(com, output)

    if args.q2:
        run_variants(
            com,
            output,
            args.q2,
            args.q2_time_limit,
            args.q2_price_bound,
            args.q2_verbose,
            args.q2_indifference
        )

    if args.q2_sensitivity:
        run_sensitivity(
            com,
            output,
            args.q2_time_limit,
            args.q2_price_bound,
            args.q2_verbose
        )

    if not (
        args.q2
        or args.q2_best_response
        or args.q2_sensitivity
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
