from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --- Paths / settings ---
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
INPUT_PATH = data_dir / "Stochastic_StochasticAvailability2.csv"

GEN_NAME = "Solar"
DECIMAL = "."
SCENARIOS = None  # None = bruk alle scenarioer i fila

COUNTRY = "Spain"
PERIOD = 1  # for enkeltland-plott

# --- Load + prep ---
df = pd.read_csv(INPUT_PATH, sep=";", decimal=DECIMAL)
df.columns = df.columns.str.strip()
df = df.rename(columns={"IntermittentGenerators": "IntermitentGenerators"})

wind = df[df["IntermitentGenerators"] == GEN_NAME].copy()
wind[["Operationalhour", "Period", "GeneratorStochasticAvailabilityRaw"]] = wind[
    ["Operationalhour", "Period", "GeneratorStochasticAvailabilityRaw"]
].apply(pd.to_numeric)

scenarios = sorted(wind["Scenario"].unique()) if SCENARIOS is None else SCENARIOS
periods = sorted(wind["Period"].unique())
hours_per_period = int(wind.groupby("Period")["Operationalhour"].max().iloc[0])
nodes = sorted(wind["Node"].unique())

# --- Stacked area: én figur per scenario, alle land, alle perioder (sammenlimt) ---
for sc in scenarios:
    sub = wind[wind["Scenario"] == sc].copy()
    sub["global_hour"] = (sub["Period"] - 1) * hours_per_period + sub["Operationalhour"]

    time_index = range(1, int(sub["global_hour"].max()) + 1)

    stacked = [
        sub[sub["Node"] == node]
        .set_index("global_hour")["GeneratorStochasticAvailabilityRaw"]
        .reindex(time_index)
        .fillna(0.0)
        .values
        for node in nodes
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(time_index, stacked, labels=nodes, alpha=0.9)

    # skygge annenhver periode + skillelinjer
    for p in periods:
        start = (p - 1) * hours_per_period + 1
        end = p * hours_per_period
        if p % 2 == 0:
            ax.axvspan(start, end, color="grey", alpha=0.15, zorder=0)
    for p in periods[1:]:
        ax.axvline((p - 1) * hours_per_period, color="k", linewidth=0.5, alpha=0.3)

    ax.set_title(f"{GEN_NAME} availability – stacked by country ({sc})")
    ax.set_xlabel("Operational hour (concatenated periods)")
    ax.set_ylabel("Stochastic availability (sum across nodes)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, frameon=False)
    plt.tight_layout()
    plt.show()

# --- Spain: én periode, én figur per scenario ---
for sc in scenarios:
    sp = wind[
        (wind["Scenario"] == sc) & (wind["Node"] == COUNTRY) & (wind["Period"] == PERIOD)
    ].sort_values("Operationalhour")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        sp["Operationalhour"],
        sp["GeneratorStochasticAvailabilityRaw"],
        alpha=0.8,
        step="mid",
    )
    ax.set_title(f"{GEN_NAME} availability – {COUNTRY}, period {PERIOD} ({sc})")
    ax.set_xlabel("Operational hour")
    ax.set_ylabel("Stochastic availability")
    plt.tight_layout()
    plt.show()
