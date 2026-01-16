from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
INPUT_PATH = data_dir / "Stochastic_StochasticAvailability2.csv"
H2_demand_data= data_dir / 'Transport.xlsx'
H2_demand_dataflex= data_dir / 'Results_FINAL_BASE_FLEX_emcap_cyclelim' / 'full_model_base' / 'results_hydrogen_production.csv'
Gen_inv_data=data_dir / 'Results_FINAL_BASE_FLEX_emcap_cyclelim' / 'full_model_base' / 'results_elec_generation_operational.csv'


Solar = "Solar"
Onshorewind='Windonshore'
DECIMAL = "."
SCENARIOS = None
COUNTRY = "Italy"
PERIOD = 6

H2_demand= pd.read_excel(H2_demand_data, sheet_name='HydrogenDemandProfile',decimal=',',skiprows=2)
H2_demand["H2_norm"] = (H2_demand["HydrogenDemand"] /H2_demand.groupby(["Node", "Period"])["HydrogenDemand"].transform("max"))

H2_demand_flex=pd.read_csv(H2_demand_dataflex,sep=',',decimal='.')
H2_demand_flex['H2_norm']=(H2_demand_flex['PEM_green production [ton]'] /H2_demand_flex.groupby(["Node",'Scenario', "Period"])["PEM_green production [ton]"].transform("max"))
H2_demand_flex["Period"] = H2_demand_flex["Period"].astype(str).str.strip()
H2_demand_flex = H2_demand_flex.rename(columns={"Hour": "Operationalhour"})

period_labels = sorted(H2_demand_flex["Period"].unique())  # sorterer på startår
period_map = {lab: i+1 for i, lab in enumerate(period_labels)}
H2_demand_flex["Period"] = H2_demand_flex["Period"].map(period_map).astype(int)

print(period_map)                 # sjekk mapping
print(H2_demand_flex["Period"].unique())


df = pd.read_csv(INPUT_PATH, sep=";", decimal=DECIMAL)
df.columns = df.columns.str.strip()
df = df.rename(columns={"IntermittentGenerators": "IntermitentGenerators"})

wind = df[df["IntermitentGenerators"] == Onshorewind].copy()
wind[["Operationalhour", "Period", "GeneratorStochasticAvailabilityRaw"]] = wind[["Operationalhour", "Period", "GeneratorStochasticAvailabilityRaw"]].apply(pd.to_numeric)

solar = df[df["IntermitentGenerators"] == Solar].copy()
solar[["Operationalhour", "Period", "GeneratorStochasticAvailabilityRaw"]] = solar[["Operationalhour", "Period", "GeneratorStochasticAvailabilityRaw"]].apply(pd.to_numeric)

scenarios = sorted(wind["Scenario"].unique()) if SCENARIOS is None else SCENARIOS
periods = sorted(wind["Period"].unique())
hours_per_period = int(wind.groupby("Period")["Operationalhour"].max().iloc[0])
nodes = sorted(wind["Node"].unique())

#Wind stacked countries:
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

    for p in periods:
        start = (p - 1) * hours_per_period + 1
        end = p * hours_per_period
        if p % 2 == 0:
            ax.axvspan(start, end, color="grey", alpha=0.15, zorder=0)
    for p in periods[1:]:
        ax.axvline((p - 1) * hours_per_period, color="k", linewidth=0.5, alpha=0.3)

    ax.set_title(f"{Onshorewind} availability – stacked by country ({sc})")
    ax.set_xlabel("Operational hour (concatenated periods)")
    ax.set_ylabel("Stochastic availability (sum across nodes)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, frameon=False)
    plt.tight_layout()
    plt.show()

#Wind:
for sc in scenarios:

    h2_flex= H2_demand_flex[
        (H2_demand_flex["Scenario"] == sc) & (H2_demand_flex["Node"] == COUNTRY) & (H2_demand_flex["Period"] == PERIOD)
    ].sort_values('Operationalhour')

    h2 = H2_demand[
        (H2_demand["Node"] == COUNTRY) & (H2_demand["Period"] == PERIOD)
    ].sort_values("Operationalhour")

    sp = wind[
        (wind["Scenario"] == sc) & (wind["Node"] == COUNTRY) & (wind["Period"] == PERIOD)
    ].sort_values("Operationalhour")

    sol = solar[
        (solar["Scenario"] == sc) & (solar["Node"] == COUNTRY) & (solar["Period"] == PERIOD)
    ].sort_values("Operationalhour")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sp["Operationalhour"], sp["GeneratorStochasticAvailabilityRaw"], label="Onshore wind",color='steelblue',linestyle='dashed')
    ax.plot(sol["Operationalhour"], sol["GeneratorStochasticAvailabilityRaw"], label="Solar",color='violet',linestyle='dashed')
    ax.plot(h2["Operationalhour"], h2["H2_norm"], label="H2 demand - BCONSTANT", linewidth=1.5,color='seagreen')
    ax.plot(h2_flex["Operationalhour"], h2_flex["H2_norm"], label="H2 production PEM green - BFLEX", linewidth=1.5,color='darkgrey',alpha=0.7)

    ax.set_xlabel("Operational hour", fontsize=18)
    ax.set_ylabel("Capacity factor", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=13,loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()


#Solar:
for sc in scenarios:
    sp = solar[
        (solar["Scenario"] == sc) & (solar["Node"] == COUNTRY) & (solar["Period"] == PERIOD)
    ].sort_values("Operationalhour")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        sp["Operationalhour"],
        sp["GeneratorStochasticAvailabilityRaw"],
        alpha=0.8,
        step="mid",
    )
    ax.set_title(f"{Solar} availability – {COUNTRY}, period {PERIOD} ({sc})")
    ax.set_xlabel("Operational hour")
    ax.set_ylabel("Stochastic availability")
    plt.tight_layout()
    plt.show()


from scipy.stats import pearsonr, spearmanr

def build_df(sc, COUNTRY, PERIOD, H2_demand, wind, solar):
    h2 = H2_demand[(H2_demand["Node"] == COUNTRY) & (H2_demand["Scenario"] == sc) & (H2_demand["Period"] == PERIOD)] \
        [["Operationalhour", "H2_norm"]].copy()

    w = wind[(wind["Scenario"] == sc) & (wind["Node"] == COUNTRY) & (wind["Period"] == PERIOD)] \
        [["Operationalhour", "GeneratorStochasticAvailabilityRaw"]].copy() \
        .rename(columns={"GeneratorStochasticAvailabilityRaw": "wind_cf"})

    s = solar[(solar["Scenario"] == sc) & (solar["Node"] == COUNTRY) & (solar["Period"] == PERIOD)] \
        [["Operationalhour", "GeneratorStochasticAvailabilityRaw"]].copy() \
        .rename(columns={"GeneratorStochasticAvailabilityRaw": "solar_cf"})

    df = h2.merge(w, on="Operationalhour", how="inner").merge(s, on="Operationalhour", how="inner").dropna()
    return df

def hexbin_with_fit(ax, x, y, xlabel, ylabel, title):
    hb = ax.hexbin(x, y, gridsize=45, mincnt=1)

    # lineær trend (om du vil bruke den)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    # ax.plot(xs, b1*xs + b0, linewidth=2)

    r, p = pearsonr(x, y)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    # tick-label størrelse per akse
    ax.tick_params(axis="both", labelsize=15)

    # grid per akse (dette var problemet)
    ax.grid(alpha=0.7)

    ax.set_title(f"{title}\nPearson r={r:.2f}")
    return hb


import matplotlib.pyplot as plt

for sc in scenarios:
    df = build_df(sc, COUNTRY, PERIOD, H2_demand_flex, wind, solar)

    # Vind: egen figur
    fig, ax = plt.subplots(figsize=(6, 5))
    hexbin_with_fit(
        ax, df["wind_cf"].values, df["H2_norm"].values,
        "Wind CF", "H2 production PEM green", f"{COUNTRY} P{PERIOD} ({sc})"
    )
    plt.show()



countries9 = ["Italy", "Portugal", "Greece", "Spain", "Belgium", "Poland", "Bulgaria", "Ireland", "Hungary"]  # <- bytt til dine 9

for sc in scenarios:
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, COUNTRY in enumerate(countries9):
        ax = axes[i]
        df = build_df(sc, COUNTRY, PERIOD, H2_demand_flex, wind, solar)

        if df.empty:
            ax.set_title(f"{COUNTRY}\nNo data")
            ax.axis("off")
            continue

        hexbin_with_fit(
            ax,
            df["solar_cf"].values,
            df["H2_norm"].values,
            "Solar CF" if i // 3 == 2 else "",              # x-label kun nederste rad
            "H2 production PEM green" if i % 3 == 0 else "", # y-label kun venstre kolonne
            f"{COUNTRY} P{PERIOD} {sc}"
        )

    # Hvis lista ikke er nøyaktig 9 (eller for sikkerhets skyld), slå av ev. tomme akser:
    for j in range(len(countries9), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

countries9 = ["Italy", "Portugal", "Greece", "Spain", "Belgium", "Poland", "Bulgaria", "Ireland", "Hungary"]

def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, len(x)
    r, p = pearsonr(x, y)
    return r, p, len(x)

for sc in scenarios:
    solar_rows = []
    wind_rows  = []

    for COUNTRY in countries9:
        df = build_df(sc, COUNTRY, PERIOD, H2_demand_flex, wind, solar)

        if df.empty:
            solar_rows.append({"Country": COUNTRY, "n": 0, "r": np.nan, "p": np.nan})
            wind_rows.append({"Country": COUNTRY, "n": 0, "r": np.nan, "p": np.nan})
            continue

        r_s, p_s, n_s = pearson_safe(df["solar_cf"].values, df["H2_norm"].values)
        r_w, p_w, n_w = pearson_safe(df["wind_cf"].values,  df["H2_norm"].values)

        solar_rows.append({"Country": COUNTRY, "n": n_s, "r": r_s, "p": p_s})
        wind_rows.append({"Country": COUNTRY, "n": n_w, "r": r_w, "p": p_w})

    solar_tab = pd.DataFrame(solar_rows).set_index("Country").round(3)
    wind_tab  = pd.DataFrame(wind_rows).set_index("Country").round(3)

    print(f"\n=== Scenario: {sc} | SOLAR vs H2_norm ===")
    print(solar_tab.to_string())

    print(f"\n=== Scenario: {sc} | WIND vs H2_norm ===")
    print(wind_tab.to_string())

