
from src.data_loader import load_community
from src.model import FlexibleConsumerModel
from src.q2_dynamic_pricing import build_dp, solve_dp, solve_central


# ============================================================
# STEP 1: LOAD COMMUNITY AND REFERENCE COSTS
# ============================================================

print("\n--- STEP 1: Loading community and benchmarks ---")

com = load_community("base", day="d1")

# Calculate individual reference costs (Model R)
R = {}

for d in com.members:
    model = FlexibleConsumerModel(d)
    R[d.consumer_id] = model.build().solve().objective

print(f"Reference cost (R): {sum(R.values()):.4f} DKK")

# Solve centralized Model C for comparison
C = solve_central(com)

C_cost = C["cost"]

print(f"Centralized cost (C): {C_cost:.4f} DKK")


# ============================================================
# STEP 2: SOLVE THE KNOWN FEASIBLE BB MODEL
# ============================================================

print("\n--- STEP 2: Solving BB starting model ---")

bb_bundle = build_dp(
    com,
    variant="bb",
    R=R
)

bb_result = solve_dp(
    bb_bundle,
    time_limit=60
)

if not bb_result["has_solution"]:
    raise RuntimeError(
        "No feasible BB starting solution found."
    )

print(f"BB status: {bb_result['status']}")

print(
    f"BB welfare cost: "
    f"{bb_result['objective_DKK']:.4f} DKK"
)

print(
    f"BB budget surplus: "
    f"{bb_result['budget_surplus_DKK']:.6f} DKK"
)

# Store solution values for warm-starting
starts = {
    v.VarName: v.X
    for v in bb_bundle["model"].getVars()
}


# ============================================================
# STEP 3: BUILD BB + RELAXED IR MODEL
# ============================================================

print("\n--- STEP 3: Building BB + IR relaxation ---")

bundle = build_dp(
    com,
    variant="bb_ir_relax",
    R=R,
    min_relax=True,
    verbose=True
)

model = bundle["model"]

# Warm-start using the feasible BB solution
for v in model.getVars():
    if v.VarName in starts:
        v.Start = starts[v.VarName]

# Obtain individual costs from the BB solution
costs = bb_result["members"].set_index(
    "consumer_id"
)["cost_DKK"]

# Initialize IR relaxation variables
for i, v in bundle["relax"].items():

    initial_relaxation = max(
        0.0,
        float(costs[i]) - R[i]
    )

    v.Start = initial_relaxation + 1e-5

# Prioritize finding feasible solutions
model.Params.MIPFocus = 1


# ============================================================
# STEP 4: MINIMIZE TOTAL IR RELAXATION
# ============================================================

print("\n--- STEP 4: Solving IR relaxation ---")

result = solve_dp(
    bundle,
    time_limit=180,
    verbose=True
)


# ============================================================
# STEP 5: PRINT RESULTS
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 2 - IR RELAXATION RESULTS")
print("=" * 55)

print(f"\nSolver status: {result['status']}")

if result["has_solution"]:

    # --------------------------------------------------------
    # 5A: Economic performance
    # --------------------------------------------------------

    print("\n--- ECONOMIC SUMMARY ---")

    print(
        f"Centralized welfare cost (C): "
        f"{C_cost:.4f} DKK"
    )

    print(
        f"Relaxed model welfare cost: "
        f"{result['objective_DKK']:.4f} DKK"
    )

    efficiency_loss = (
        result["objective_DKK"] - C_cost
    )

    print(
        f"Efficiency loss versus C: "
        f"{efficiency_loss:.4f} DKK"
    )

    print(
        f"Manager budget surplus: "
        f"{result['budget_surplus_DKK']:.6f} DKK"
    )

    print(
        f"Total cap exceedance: "
        f"{result['cap_exceedance_kWh']:.4f} kWh"
    )

    print(
        f"Peak community import: "
        f"{result['peak_import_kW']:.4f} kW"
    )

    # --------------------------------------------------------
    # 5B: IR relaxation
    # --------------------------------------------------------

    print("\n--- INDIVIDUAL RATIONALITY ---")

    total_relaxation = result["total_IR_relax_DKK"]

    print(
        f"Total IR relaxation: "
        f"{total_relaxation:.6f} DKK"
    )

    df = result["members"][
        [
            "consumer_id",
            "cost_DKK",
            "IR_relax_DKK"
        ]
    ].copy()

    df["reference_R_DKK"] = df["consumer_id"].map(R)

    df["gain_vs_R_DKK"] = (
        df["reference_R_DKK"] - df["cost_DKK"]
    )

    df = df[
        [
            "consumer_id",
            "reference_R_DKK",
            "cost_DKK",
            "gain_vs_R_DKK",
            "IR_relax_DKK"
        ]
    ]

    print("\nResults for all 16 members:")

    print(
        df.round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 5C: Identify members requiring relaxation
    # --------------------------------------------------------

    blocking = df[
        df["IR_relax_DKK"] > 1e-5
    ].copy()

    print("\n--- MEMBERS REQUIRING IR RELAXATION ---")

    print(
        f"Number of members: {len(blocking)}"
    )

    if len(blocking) > 0:

        print(
            blocking[
                [
                    "consumer_id",
                    "IR_relax_DKK"
                ]
            ].round(4).to_string(index=False)
        )

    else:
        print("No members require IR relaxation.")

    # --------------------------------------------------------
    # 5D: Numerical validation
    # --------------------------------------------------------

    print("\n--- NUMERICAL VALIDATION ---")

    print(
        f"Maximum payment residual: "
        f"{result['max_payment_gap']:.2e}"
    )

    print(
        f"Maximum stationarity residual: "
        f"{result['max_stationarity']:.2e}"
    )

    print(
        f"Maximum complementarity residual: "
        f"{result['max_complementarity']:.2e}"
    )

    print(
        f"MIP gap: {result['mip_gap']}"
    )

    # --------------------------------------------------------
    # 5E: Interpretation
    # --------------------------------------------------------

    print("\n--- CONCLUSION ---")

    if result["status"] == "OPTIMAL":

        if total_relaxation <= 1e-5:

            print(
                "BB + IR is feasible: "
                "the minimum required IR relaxation is zero."
            )

        else:

            print(
                "BB + IR is infeasible under the modeled "
                "uniform-price mechanism."
            )

            print(
                "Minimum required total IR relaxation: "
                f"{total_relaxation:.6f} DKK"
            )

    else:

        print(
            "A feasible relaxed solution was found, "
            "but global optimality was not proven."
        )

        print(
            "The reported IR relaxation is an UPPER BOUND "
            "on the minimum required relaxation."
        )

        print(
            "The feasibility of strict BB + IR "
            "remains unresolved."
        )

else:

    print(
        "\nNo feasible solution found within "
        "the solver time limit."
    )

    print(
        "This does not prove that the model "
        "is infeasible."
    )

print("\n" + "=" * 55)
print("END OF IR RELAXATION ANALYSIS")
print("=" * 55)
