from pathlib import Path
import pandas as pd

from functions_plot import (
    Expected_annual_production,
    plot_top_map,
    P_prodVSimport_piechart,
    Plot_Installed_capacity_per_tech_split,
    HydrogenProd_piechart,
    plot_power_demand,
    plot_hydrogen_use,
    plot_OBJ_generator_inv,
    plot_H2_costs_per_period,
    plot_DGF_charge_discharge_stochastic,
    HydrogenStorage_scatter,
    plot_discharge_cycles_sawtooth,
    plot_power_balance_for_high_h2,
    hydrogen_prod_vs_import_bar,
    Yearly_hydrogenProd_perTech,
    plot_storage_charge_discharge_total,
    plot_h2_prod_discharge_plus_demand,
    Power_gen_hourly,
    Duration_curve_power_prod,
    plot_transmission_utilization_duration_curve,
    plot_capacity_factors,
    plot_top_annual_expected_generation,
)


# -----------------------
# Paths / config
# -----------------------
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
result_dir = data_dir / "Results_FINAL_optimistic_price_scen" / "full_model_base"

plot_dir = data_dir / "Results_woH2_storage"
plot_dir.mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = False
FIGURE_TAG = "BASE_moderate"

DO_GEN_INFO = False
DO_H2_PROD = False
DO_EL_DEMAND = False
DO_H2_STORAGE = True
DO_POWER_OPERATIONAL= False

# -----------------------
# General info
# -----------------------
if DO_GEN_INFO:
    elec_generation_inv = pd.read_csv(result_dir / "results_elec_generation_inv.csv")

    Expected_annual_production(
        elec_generation_inv,
        savefigure=SAVE_FIGURES,
        results_dir=plot_dir,
        figurename=FIGURE_TAG,
    )

    europe_summary = pd.read_csv(
        result_dir / "results_output_EuropeSummary.csv",
        delimiter=",",
        skiprows=16,
        usecols=[0, 1, 2, 3, 4, 5],
        skipfooter=16,
        engine="python",
    )
    europe_summary["genExistingCap_MW"] = (
        europe_summary["genInstalledCap_MW"] - europe_summary["genInvCap_MW"]
    )

    Plot_Installed_capacity_per_tech_split(
        europe_summary,
        threshold=90_000,
        figsize=(18, 12),
        savefigure=SAVE_FIGURES,
        figurename=FIGURE_TAG,
        results_dir=plot_dir,
    )

if DO_POWER_OPERATIONAL:

    power_generation = pd.read_csv(result_dir / "results_elec_generation_operational.csv", index_col=False)
    Power_gen_hourly(power_generation)
    Duration_curve_power_prod(power_generation)

    transmission_operational = pd.read_csv(result_dir / "results_elec_transmission_operational.csv")
    transmission_inv = pd.read_csv(result_dir / "results_transmission_inv.csv")

    plot_transmission_utilization_duration_curve(
        transmission_operational,
        transmission_inv,
        scenario="scenario2",
        link="France - Italy",
        savefigure=SAVE_FIGURES,
        results_dir=plot_dir,
        figurename_prefix=FIGURE_TAG,
        show=True,
    )


# -----------------------
# Power balance + maps
# -----------------------
power_balance = pd.read_csv(result_dir / "results_power_balance.csv")

if DO_GEN_INFO:
    P_prodVSimport_piechart(
        power_balance, 12, 2,
        savefigure=SAVE_FIGURES,
        figurename=FIGURE_TAG,
        results_dir=plot_dir,
    )

    plot_top_map(
        elec_generation_inv, 3,
        savefigure=SAVE_FIGURES,
        figurename=FIGURE_TAG,
        results_dir=plot_dir,
    )


# -----------------------
# Hydrogen production
# -----------------------
hydrogen_production = pd.read_csv(result_dir / "results_hydrogen_production.csv")

if DO_H2_PROD:
    HydrogenProd_piechart(
        hydrogen_production, 12, 2,
        savefigure=SAVE_FIGURES,
        figurename=FIGURE_TAG,
        results_dir=plot_dir,
    )

    Yearly_hydrogenProd_perTech(
        hydrogen_production,
        "PEM_green production [ton]",
        "PEM_yellow production [ton]",
        "PEM_import production [ton]",
        "ALK production [ton]",
        "SOEC production [ton]",
        "Reformer production [ton]",
        2, 12, 55,
    )


# -----------------------
# Electricity demand
# -----------------------
if DO_EL_DEMAND:
    power_balance = power_balance.copy()
    power_balance["Power reformer plant [MWh]"] *= -1
    plot_power_demand(power_balance, n_hours=12)


# -----------------------
# Hydrogen use
# -----------------------
hydrogen_use = pd.read_csv(result_dir / "results_hydrogen_use.csv")

if DO_H2_PROD:
    plot_hydrogen_use(hydrogen_use, 12, 2, savefigure=False, figurename=None, results_dir=None)
    hydrogen_prod_vs_import_bar(hydrogen_use)


# -----------------------
# Hydrogen storage / costs
# -----------------------
if DO_H2_STORAGE:
    gen_inv = pd.read_csv(result_dir / "results_objective_components_generation_inv_costs.csv")
    gen_op = pd.read_csv(result_dir / "results_objective_components_operational_costs.csv")
    h2_inv = pd.read_csv(result_dir / "results_hydrogen_costs.csv")
    stor_el = pd.read_csv(result_dir / "results_objective_components_storage_inv_costs.csv")
    trans_inv_costs = pd.read_csv(result_dir / "results_objective_components_transmission_inv_costs.csv")

    plot_OBJ_generator_inv(gen_op, gen_inv, h2_inv, stor_el, trans_inv_costs, 2)
    plot_H2_costs_per_period(h2_inv)

    h2_storage_inv = pd.read_csv(result_dir / "results_hydrogen_storage_inv.csv")
    periods = [
        "2020-2025", "2025-2030", "2030-2035", "2035-2040",
        "2040-2045", "2045-2050", "2050-2055",
    ]
    for period in periods:
        HydrogenStorage_scatter(h2_storage_inv, period, savefigure=False, figurename=None, results_dir=None)

    h2_storage_op = pd.read_csv(result_dir / "results_hydrogen_storage_operational.csv")

    plot_DGF_charge_discharge_stochastic(
        h2_storage_op,
        node="Germany",
        period="2045-2050",
        gasscenario=1,
        scenario="scenario1",
    )

    plot_discharge_cycles_sawtooth(
        h2_storage_op,
        node="Germany",
        period="2045-2050",
        gasscenario=1,
        tech="DGF",
        capacity=6545.9,
        scenario="scenario1",
    )

    plot_power_balance_for_high_h2(power_balance, "2045-2050", "scenario1", "Germany")

    for tech in ["DGF", "Aquifer", "SaltCavern"]:
        plot_storage_charge_discharge_total(h2_storage_op, "2045-2050", 1, "scenario1", tech)

    _merged = plot_h2_prod_discharge_plus_demand(
        hydrogen_production,
        h2_storage_op,
        hydrogen_use,
        period="2050-2055",
        gasscenario=1,
        scenario="scenario2",  # eller None for snitt
    )


# -----------------------
# NEW: capacity factors + top expected generation (moved into functions_plots)
# -----------------------
_ = plot_capacity_factors(
    result_dir,
    savefigure=SAVE_FIGURES,
    results_dir=plot_dir,
    figurename_prefix=FIGURE_TAG,
    show=True,
)

plot_top_annual_expected_generation(
    result_dir,
    n_top=5,
    group_by="GeneratorType",
    show=True,
)


# --------------------------------
# Kjernekraft europa
# -------------------------------

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.patches import Patch

def plot_europe_colored_countries(savefigure=False, figurename="europe_colored.png"):
    # --- Landgrupper (med noen vanlige alias) ---
    teal = {
        "Belarus",               # (du skrev Belsarus)
        "Belgium",
        "Bulgaria",
        "Czechia", "Czech Republic",
        "Finland",
        "France",
        "Hungary",
        "Netherlands",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
        "United Kingdom",
        "Ukraine",
    }

    grey = {"Norway", "Lithuania", "Latvia"}  # spesifisert av deg (men alle "andre" blir også grå)
    yellow = {"Estonia", "Poland"}
    orange_red = {'Ireland',"Italy", "Portugal", "Germany", "Luxembourg", "Austria", "Denmark"}

    # Farger
    COLORS = {
        "teal": "teal",
        "grey": "lightgrey",
        "gold": "gold",
        "orange": "orange",
    }

    # --- Kartoppsett (likt som i din funksjon) ---
    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-11, 40, 34, 72], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.6, linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    # --- Landpolygoner (Natural Earth) ---
    shp = shapereader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_0_countries"
    )
    reader = shapereader.Reader(shp)

    # Default: alle europeiske land = grå, med overstyring for gruppene over
    for rec in reader.records():
        attrs = rec.attributes
        continent = attrs.get("CONTINENT", "")
        name = attrs.get("ADMIN", "")

        if continent != "Europe":
            continue

        if name in teal:
            fc = COLORS["teal"]
        elif name in yellow:
            fc = COLORS["gold"]
        elif name in orange_red:
            fc = COLORS["orange"]
        else:
            fc = COLORS["grey"]  # alt annet i Europa blir grått (inkl. land du ikke nevnte)

        ax.add_geometries(
            [rec.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=fc,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.95
        )

    # Legende
    legend_handles = [
        Patch(facecolor=COLORS["teal"], edgecolor="black", label="Operational NPP plants"),
        Patch(facecolor=COLORS["grey"], edgecolor="black", label="No NPP generation per date."),
        Patch(facecolor=COLORS["gold"], edgecolor="black", label="No NPP per date, but planned construction."),
        Patch(facecolor=COLORS["orange"], edgecolor="black", label="No NPP, and bans building per date."),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=18)

    plt.tight_layout()

    if savefigure:
        plt.savefig(figurename, dpi=300, bbox_inches="tight")
        print(f"Saved: {figurename}")

    plt.show()

# Kjør:
plot_europe_colored_countries()
