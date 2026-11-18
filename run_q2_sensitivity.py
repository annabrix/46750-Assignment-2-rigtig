
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_community
from src.model import FlexibleConsumerModel
from src.community_model import (
    solve_central,
    build_dp,
    solve_dp,
)


# ============================================================
# SETTINGS
# ============================================================

COMMUNITY = "base"
DAY = "d1"

# Start with two discounts to validate the experiment.
BETAS = [0, 0.25, 0.50, 0.625, 0.75, 0.875, 1.0]

# Maximum time per dynamic pricing model.
TIME_LIMIT = 60

VARIANTS = ["cap", "bb"]

OUT_DIR = Path("results") / COMMUNITY / DAY / "Q2"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: REFERENCE BENCHMARK
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 2 - TARIFF DISCOUNT SENSITIVITY")
print("=" * 60)

com = load_community(COMMUNITY, day=DAY)

R = {}

for member in com.members:
    model = FlexibleConsumerModel(member)
    R[member.consumer_id] = model.build().solve().objective

R_total = sum(R.values())

print(f"\nReference R cost: {R_total:.4f} DKK")


# ============================================================
# STEP 2: RUN SENSITIVITY
# ============================================================

rows = []

for beta in BETAS:

    print("\n" + "-" * 60)
    print(f"DSO tariff discount beta = {beta:.3f}")
    print("-" * 60)

    # Centralized optimum must be recomputed for each beta.
    C = solve_central(com, discount=beta)
    C_cost = float(C["cost"])

    print(f"Centralized cost C: {C_cost:.4f} DKK")
    print(f"Reference cost R:   {R_total:.4f} DKK")

    aggregate_saving = R_total - C_cost

    # Necessary condition for simultaneous BB and IR.
    bb_ir_analytically_infeasible = (
        C_cost > R_total + 1e-5
    )

    if bb_ir_analytically_infeasible:
        print("BB+IR: analytically infeasible.")
    else:
        print("BB+IR: necessary aggregate condition satisfied.")
        print("Full feasibility is not yet established.")

    # Check that the base-case benchmark is consistent.
    if abs(beta - 0.75) < 1e-9:
        if abs(C_cost - 372.1624) > 0.01:
            raise RuntimeError(
                "Central benchmark does not match the base case."
            )

    # Solve cap-only and budget-balanced pricing.
    for variant in VARIANTS:

        print(f"\nSolving DP {variant} ...", flush=True)

        bundle = build_dp(
            com,
            variant=variant,
            R=R,
            discount=beta,
        )

        result = solve_dp(
            bundle,
            time_limit=TIME_LIMIT,
            verbose=False,
        )

        has_solution = result["has_solution"]
        status = result["status"]

        row = {
            "beta": beta,
            "variant": variant,
            "status": status,
            "C_cost_DKK": C_cost,
            "R_cost_DKK": R_total,
            "aggregate_saving_C_DKK": aggregate_saving,
            "BB_IR_analytically_infeasible":
                bb_ir_analytically_infeasible,
            "DP_cost_DKK": np.nan,
            "efficiency_loss_DKK": np.nan,
            "budget_surplus_DKK": np.nan,
            "cap_exceedance_kWh": np.nan,
            "peak_import_kW": np.nan,
            "IR_violations_R": np.nan,
            "blocking_members": "",
            "mip_gap": result.get("mip_gap", np.nan),
        }

        if has_solution:

            DP_cost = float(result["objective_DKK"])
            efficiency_loss = DP_cost - C_cost

            if abs(efficiency_loss) < 1e-5:
                efficiency_loss = 0.0

            members_df = result["members"].copy()

            members_df["R_cost_DKK"] = (
                members_df["consumer_id"].map(R)
            )

            members_df["gain_vs_R_DKK"] = (
                members_df["R_cost_DKK"]
                - members_df["cost_DKK"]
            )

            losing = members_df[
                members_df["gain_vs_R_DKK"] < -1e-5
            ]

            row.update({
                "DP_cost_DKK": DP_cost,
                "efficiency_loss_DKK": efficiency_loss,
                "budget_surplus_DKK":
                    result["budget_surplus_DKK"],
                "cap_exceedance_kWh":
                    result["cap_exceedance_kWh"],
                "peak_import_kW":
                    result["peak_import_kW"],
                "IR_violations_R": len(losing),
                "blocking_members": ",".join(
                    losing["consumer_id"].tolist()
                ),
            })

            print(f"Status: {status}")
            print(f"Welfare: {DP_cost:.4f} DKK")
            print(f"Efficiency loss: {efficiency_loss:.4f} DKK")
            print(
                f"Budget surplus: "
                f"{result['budget_surplus_DKK']:.4f} DKK"
            )
            print(f"IR violations: {len(losing)}")
            print(
                f"Cap exceedance: "
                f"{result['cap_exceedance_kWh']:.4f} kWh"
            )

        else:
            print(f"Status: {status}")
            print("No feasible incumbent was found.")

        rows.append(row)

        # Save after every solve, so results are retained.
        pd.DataFrame(rows).to_csv(
            OUT_DIR / "q2_sensitivity_beta.csv",
            index=False,
        )


# ============================================================
# STEP 3: SUMMARY TABLE
# ============================================================

df = pd.DataFrame(rows)

print("\n" + "=" * 60)
print("SENSITIVITY RESULTS")
print("=" * 60)

columns = [
    "beta",
    "variant",
    "status",
    "C_cost_DKK",
    "DP_cost_DKK",
    "efficiency_loss_DKK",
    "budget_surplus_DKK",
    "IR_violations_R",
    "cap_exceedance_kWh",
]

print(df[columns].round(4).to_string(index=False))


# ============================================================
# STEP 4: EFFICIENCY LOSS FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))

for variant, label in [
    ("cap", "DP cap"),
    ("bb", "DP BB"),
]:

    subset = df[
        (df["variant"] == variant)
        & (df["status"] == "OPTIMAL")
    ].sort_values("beta")

    ax.plot(
        subset["beta"],
        subset["efficiency_loss_DKK"],
        marker="o",
        label=label,
    )

ax.axhline(0, linestyle="--", color="gray")

ax.set_xlabel(r"DSO tariff discount $\beta^{DSO}$")
ax.set_ylabel("Efficiency loss [DKK/day]")
ax.set_title("Dynamic pricing efficiency loss")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_q2_sensitivity.pdf")
plt.close(fig)


# ============================================================
# STEP 5: INDIVIDUAL RATIONALITY FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))

for variant, label in [
    ("cap", "DP cap"),
    ("bb", "DP BB"),
]:

    subset = df[
        (df["variant"] == variant)
        & (df["status"] == "OPTIMAL")
    ].sort_values("beta")

    ax.plot(
        subset["beta"],
        subset["IR_violations_R"],
        marker="o",
        label=label,
    )

ax.set_xlabel(r"DSO tariff discount $\beta^{DSO}$")
ax.set_ylabel("Members worse off than R")
ax.set_title("Individual rationality sensitivity")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_q2_ir_sensitivity.pdf")
plt.close(fig)


print("\nResults saved in:", OUT_DIR)
print("Figures saved in:", FIG_DIR)
print("\nSensitivity analysis completed.")
