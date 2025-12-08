from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir1= data_dir / 'Results_BASE_MOD_NOFLEX' / 'full_model_base'
result_dir2= data_dir / 'Results_BASE_MOD_FLEX' / 'full_model_base'


def compute_period_costs(Gen_op, Gen_inv, H2_inv, Stor_el, Trans_inv, n_scen):
    """
    Returnerer en DataFrame med periodiske (ikke kumulative) kostnader per komponent [bn EUR].
    """

    # Generator driftskostnad (diskontert i input, skaleres med n_scen)
    gen_op_period = (
        Gen_op.groupby("Period")["OperationalCost_Euro"].sum() / (n_scen * 10**9)
    )

    # Generator investeringskostnad
    gen_inv_period = (
        Gen_inv.groupby("Period")["genInvestedCost_Euro"].sum() / 10**9
    )

    # HYDROGEN: gjør kumulative tall om til periodiske
    h2_prod_cols = [
        'Discounted PEM_yellow cost [EUR]',
        'Discounted PEM_green cost [EUR]',
        'Discounted ALK cost [EUR]',
        'Discounted SOEC cost [EUR]',
        'Discounted Reformer cost [EUR]'
    ]

    # Kumulativ produksjonskostnad per periode
    h2_prod_cum = (
        H2_inv.groupby("Period")[h2_prod_cols]
        .sum()
        .sum(axis=1)
        .sort_index()
    )
    # Periodisk (inkrementell) produksjonskostnad
    h2_prod_period = h2_prod_cum.diff().fillna(h2_prod_cum) / 10**9

    # Rør
    h2_pipe_cum = (
        H2_inv.groupby("Period")['Discounted pipeline cost [EUR]']
        .sum()
        .sort_index()
    )
    h2_pipe_period = h2_pipe_cum.diff().fillna(h2_pipe_cum) / 10**9

    # Lagring
    h2_stor_cum = (
        H2_inv.groupby("Period")['Discounted storage cost [EUR]']
        .sum()
        .sort_index()
    )
    h2_stor_period = h2_stor_cum.diff().fillna(h2_stor_cum) / 10**9

    # Power storage & transmission
    pstor_period = (
        Stor_el.groupby("Period")['storInvestedCost_Euro'].sum() / 10**9
    )

    ptrans_period = (
        Trans_inv.groupby("Period")['transmissionInvestedCost_Euro'].sum() / 10**9
    )

    df = pd.DataFrame({
        "Generator operational": gen_op_period,
        "Generator investment": gen_inv_period,
        "Hydrogen production inv.": h2_prod_period,
        "Hydrogen pipeline inv.": h2_pipe_period,
        "Hydrogen storage inv.": h2_stor_period,
        "Power storage inv.": pstor_period,
        "Power transmission inv.": ptrans_period
    })

    return df

def plot_OBJ_generator_inv_diff(df_base, df_scen,
                                name_base="BASE_MOD_NOFLEX",
                                name_scen="BASE_MOD_FLEX"):
    """
    Plotter forskjellen i objektfunksjonskomponenter mellom et scenario og en base.
    df_base og df_scen er output fra compute_period_costs().
    """

    periods = ["2020-2025", "2025-2030", "2030-2035", "2035-2040",
               "2040-2045", "2045-2050", "2050-2055"]
    n_periods = len(periods)

    # Juster begge til samme periodeindeks
    df_base = df_base.reindex(periods).fillna(0)
    df_scen = df_scen.reindex(periods).fillna(0)

    # Differanse: scenario – base
    df_diff = df_scen - df_base

    colors = [
        "teal",
        "darkturquoise",
        "orange",
        "yellowgreen",
        "plum",
        "hotpink",
        "darkslategrey"
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(n_periods)
    bottom_pos = np.zeros(n_periods)  # for positive bidrag
    bottom_neg = np.zeros(n_periods)  # for negative bidrag

    for col, color in zip(df_diff.columns, colors):
        vals = df_diff[col].values

        # Del opp i positive og negative komponenter så stacking gir mening
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)

        # Positive stacked oppover
        ax.bar(x, pos, bottom=bottom_pos, color=color, label=col)
        bottom_pos += pos

        # Negative stacked nedover
        ax.bar(x, neg, bottom=bottom_neg, color=color)
        bottom_neg += neg

    # Null-linje (basen)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha="right")
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_ylim([-130, 130])
    ax.set_xlabel("Period", fontsize=16)
    ax.set_ylabel(f"Difference vs {name_base} [bn EUR]", fontsize=16)
    ax.set_title(f"{name_scen} – {name_base}: objective cost components", fontsize=16)
    ax.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()


# Eksempel – du må tilpasse filnavn til dine faktiske CSV-navn
def load_results(result_dir):
    Gen_op   = pd.read_csv(result_dir / 'results_objective_components_operational_costs.csv')
    Gen_inv  = pd.read_csv(result_dir / "results_objective_components_generation_inv_costs.csv")
    H2_inv   = pd.read_csv(result_dir / "results_hydrogen_costs.csv")
    Stor_el  = pd.read_csv(result_dir / "results_objective_components_storage_inv_costs.csv")
    Trans_inv= pd.read_csv(result_dir / "results_objective_components_transmission_inv_costs.csv")
    return Gen_op, Gen_inv, H2_inv, Stor_el, Trans_inv

n_scen = 2  # eller hva du faktisk har

Gen_op1, Gen_inv1, H2_inv1, Stor_el1, Trans_inv1 = load_results(result_dir1)
Gen_op2, Gen_inv2, H2_inv2, Stor_el2, Trans_inv2 = load_results(result_dir2)

df_base = compute_period_costs(Gen_op2, Gen_inv2, H2_inv2, Stor_el2, Trans_inv2, n_scen)
df_scen = compute_period_costs(Gen_op1, Gen_inv1, H2_inv1, Stor_el1, Trans_inv1, n_scen)

plot_OBJ_generator_inv_diff(df_base, df_scen,
                            name_base="BASE_MOD_FLEX",
                            name_scen="BASE_MOD_NOFLEX")

def plot_total_objective_difference(df_base, df_scen,
                                   name_base="BASE",
                                   name_scen="SCENARIO"):

    df_diff = df_scen - df_base
    comp_diff = df_diff.sum(axis=0)   # komponent -> bn EUR
    components = comp_diff.index.tolist()
    values = comp_diff.values

    fig, ax = plt.subplots(figsize=(8, 6))

    x = 0.0                  # én numerisk posisjon
    bar_width = 0.5         # smal bar
    bottom_pos = 0.0
    bottom_neg = 0.0

    colors = [
        "teal",
        "darkturquoise",
        "orange",
        "yellowgreen",
        "plum",
        "hotpink",
        "darkslategrey"
    ]

    for comp, val, col in zip(components, values, colors):
        if np.isclose(val, 0.0):
            continue

        if val > 0:
            ax.bar(x, val,
                   width=bar_width,
                   bottom=bottom_pos,
                   color=col,
                   edgecolor="black",
                   linewidth=0.5,
                   label=comp)
            bottom_pos += val
        else:
            ax.bar(x, val,
                   width=bar_width,
                   bottom=bottom_neg,
                   color=col,
                   edgecolor="black",
                   linewidth=0.5,
                   label=comp)
            bottom_neg += val

    # Nå: gjør aksen bredere enn baren
    ax.set_xlim(-1, 1)   # NØKKELEN: større x-range → visuelt tynnere stolpe
    ax.set_xticks([x])
    ax.set_xticklabels([f"{name_scen} – {name_base}"])

    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel("Difference [bn EUR]", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title="Components",
              fontsize=10, title_fontsize=11,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    return comp_diff


comp_diff = plot_total_objective_difference(
    df_base, df_scen,
    name_base="BASE_MOD_FLEX",
    name_scen="BASE_MOD_NOFLEX"
)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# --- Teknologifarger (fra deg) ---
tech_colors = {
    'Bio': 'darkslategrey',
    'Bioexisting': 'mediumaquamarine',
    'Coal': 'teal',
    'Coalexisting':'black',
    'GasCCGT': 'coral',
    'GasOCGT': 'skyblue',
    'Gasexisting': 'royalblue',
    'Geo': 'steelblue',
    'HydrogenCCGT': 'moccasin',
    'HydrogenOCGT': 'orange',
    'Hydroregulated': 'khaki',
    'Hydrorun-of-the-river': 'yellowgreen',
    'Liginiteexisting': 'maroon',
    'Lignite': 'sienna',
    'LigniteCCSadv': 'chocolate',
    'Nuclear': 'pink',
    'Oilexisting': 'purple',
    'Solar': 'violet',
    'Waste': 'navy',
    'Wave': 'darkslateblue',
    'Windoffshorefloating': 'slateblue',
    'Windoffshoregrounded': 'lightsteelblue',
    'Windonshore': 'seagreen'
}

# ---- Beregning av produksjon per teknologi og periode ----
def compute_production(df):
    work = df.copy()

    # sikre numerisk datatype
    work["genExpectedAnnualProduction_GWh"] = pd.to_numeric(
        work["genExpectedAnnualProduction_GWh"], errors="coerce"
    )

    prod = work.pivot_table(
        index="GeneratorType",
        columns="Period",
        values="genExpectedAnnualProduction_GWh",
        aggfunc="sum",
        fill_value=0
    )

    # sorter perioder etter startår
    def get_year(p):
        m = re.match(r"(\d{4})", str(p))
        return int(m.group(1)) if m else 9999

    periods = sorted(prod.columns, key=get_year)
    prod = prod[periods]

    return prod, periods


def plot_production_diff(df_base, df_scen, name_base="BASE", name_scen="SCEN"):

    prod_base, periods = compute_production(df_base)
    prod_scen, _       = compute_production(df_scen)

    # alle teknologier som finnes
    all_techs = sorted(set(prod_base.index) | set(prod_scen.index))

    prod_base = prod_base.reindex(all_techs).fillna(0)
    prod_scen = prod_scen.reindex(all_techs).fillna(0)

    # differanse
    prod_diff = prod_scen - prod_base   # GWh
    prod_diff_TWh = prod_diff / 1e3     # TWh for mer lesbar figur

    x = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(16, 10))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for tech in all_techs:
        vals = prod_diff_TWh.loc[tech].values

        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)

        color = tech_colors.get(tech, "lightgray")

        ax.bar(x, pos, bottom=bottom_pos,
               color=color, edgecolor="black", linewidth=0.3)
        bottom_pos += pos

        ax.bar(x, neg, bottom=bottom_neg,
               color=color, edgecolor="black", linewidth=0.3)
        bottom_neg += neg

    ax.axhline(0, color="black")

    ax.set_xticks(x)
    ax.set_ylim([-900,900])
    ax.set_xticklabels(periods, rotation=30, ha="right")
    ax.set_ylabel(f"Difference in expected annual production [{name_scen} – {name_base}] [TWh]", fontsize=14)
    ax.set_title(f"Annual production difference per period ({name_scen} – {name_base})", fontsize=16)

    # Legende
    handles = [plt.Rectangle((0,0),1,1,color=tech_colors.get(t,"lightgray")) for t in all_techs]
    ax.legend(handles, all_techs, title="Technology", fontsize=10, ncol=2,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return prod_diff_TWh


Elec_genInv_FLEX=pd.read_csv(result_dir2/'results_elec_generation_inv.csv')
Elec_genInv_NOFLEX=pd.read_csv(result_dir1/'results_elec_generation_inv.csv')

df_base = Elec_genInv_FLEX     # fra Results_BASE_MOD_NOFLEX
df_scen = Elec_genInv_NOFLEX     # fra Results_BASE_MOD_FLEX

cap_diff_TW = plot_production_diff(
    df_base=df_base,
    df_scen=df_scen,
    name_base="FLEX",
    name_scen="NOFLEX",
)

def Yearly_hydrogenProd_perTech_diff(
        df_base, df_scen,
        x1, x2, x3, x4, x5, x6,
        n_scen, n_hours,
        name_base="FLEX", name_scen="NOFLEX",
        y_lim=None, savefigure=False, results_dir=None, figurename=None
):


    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    periods_order = ["2020-2025", "2025-2030", "2030-2035",
                     "2035-2040", "2040-2045", "2045-2050", "2050-2055"]

    def _prep(df):
        agg = (df.groupby("Period")
               .agg({x1: "sum", x2: "sum", x3: "sum",
                     x4: "sum", x5: "sum", x6: "sum"})
               .reset_index())

        agg["Period"] = pd.Categorical(agg["Period"],
                                       categories=periods_order,
                                       ordered=True)
        agg = agg.sort_values("Period")

        # skaler slik som i original-funksjonen
        agg[[x1, x2, x3, x4, x5, x6]] = (
            agg[[x1, x2, x3, x4, x5, x6]] * seasonScale / n_scen
        )

        # sett Period som index for enklere diff
        agg = agg.set_index("Period")
        return agg

    agg_base = _prep(df_base)
    agg_scen = _prep(df_scen)

    agg_base = agg_base.reindex(periods_order).fillna(0)
    agg_scen = agg_scen.reindex(periods_order).fillna(0)

    diff = agg_scen - agg_base
    periods = diff.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = np.arange(len(periods)) * 0.5
    width = 0.3

    tech_cols = [x1, x2, x3, x4, x5, x6]
    labels = ["PEM green", "PEM yellow", "PEM import",
              "Alkaline", "SOEC", "Reformer"]
    colors = ["seagreen", "gold", "orange",
              "plum", "rebeccapurple", "grey"]

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for col, lab, color in zip(tech_cols, labels, colors):
        vals = diff[col].values / 1e6

        pos_vals = np.where(vals > 0, vals, 0.0)
        neg_vals = np.where(vals < 0, vals, 0.0)

        # positive deler
        ax.bar(pos, pos_vals, width,
               bottom=bottom_pos,
               color=color, alpha=0.8,
               edgecolor="black", linewidth=0.4,
               label=lab)
        bottom_pos += pos_vals

        # negative deler
        ax.bar(pos, neg_vals, width,
               bottom=bottom_neg,
               color=color, alpha=0.8,
               edgecolor="black", linewidth=0.4)
        bottom_neg += neg_vals

    ax.axhline(0, color="black", linewidth=1)

    ax.set_xticks(pos)
    ax.set_ylim([-12,12])
    ax.set_xticklabels(periods, rotation=0)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    ax.set_xlabel("Investment period", fontsize=18)
    ax.set_ylabel(
        f"Δ annual hydrogen production [{name_scen} – {name_base}] [M ton]",
        fontsize=18
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=16,
              title='Technology', title_fontsize=18)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if savefigure and figurename and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f'{figurename}_AnnH2prod_DIFF.png'
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {figpath}')

    plt.tight_layout()
    plt.show()

    return diff

hydrogen_production_FLEX=pd.read_csv(result_dir2/ 'results_hydrogen_production.csv')
hydrogen_production_NOFLEX=pd.read_csv(result_dir1/ 'results_hydrogen_production.csv')

diff = Yearly_hydrogenProd_perTech_diff(
    df_base=hydrogen_production_FLEX,
    df_scen=hydrogen_production_NOFLEX,
    x1='PEM_green production [ton]',
    x2='PEM_yellow production [ton]',
    x3='PEM_import production [ton]',
    x4='ALK production [ton]',
    x5='SOEC production [ton]',
    x6='Reformer production [ton]',
    n_scen=2,
    n_hours=12,
    name_base="FLEX",
    name_scen="NOFLEX"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def hydrogen_storage_capacity_diff(
        df_base,
        df_scen,
        name_base="FLEX",
        name_scen="NOFLEX",
        periods=None,
        savefigure=False,
        results_dir=None,
        figurename=None
):
    """
    Stacked difference-plot for hydrogenlagringskapasitet per periode.
    Bruker total lagringskapasitet (tonn) og viser diff = scenario - base.

    Forventer kolonner:
      'Total SaltCavern storage capacity [ton]',
      'Total DGF storage capacity [ton]',
      'Total Aquifer storage capacity [ton]'
    """

    if periods is None:
        periods = ["2020-2025", "2025-2030", "2030-2035",
                   "2035-2040", "2040-2045", "2045-2050", "2050-2055"]

    cols = [
        "Total SaltCavern storage capacity [ton]",
        "Total DGF storage capacity [ton]",
        "Total Aquifer storage capacity [ton]",
    ]
    labels = ["Salt cavern", "DGF", "Aquifer"]
    colors = ["lightskyblue", "orange", "yellowgreen"]

    def _prep(df):
        work = df.copy()
        # sikre numerisk
        for c in cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        agg = (
            work.groupby("Period")[cols]
            .sum()
            .reindex(periods)
            .fillna(0)
        )
        return agg

    cap_base = _prep(df_base)
    cap_scen = _prep(df_scen)

    # diff: scenario - base (tonn)
    diff = cap_scen - cap_base
    # til Mton
    diff_Mton = diff / 1e6

    # plotting
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(12, 7))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for c, lab, col in zip(cols, labels, colors):
        vals = diff_Mton[c].values

        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        # positive deler
        ax.bar(
            x,
            pos,
            bottom=bottom_pos,
            color=col,
            edgecolor="black",
            linewidth=0.4,
            label=lab
        )
        bottom_pos += pos

        # negative deler
        ax.bar(
            x,
            neg,
            bottom=bottom_neg,
            color=col,
            edgecolor="black",
            linewidth=0.4
        )
        bottom_neg += neg

    ax.axhline(0, color="black", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel(
        f"Hydrogen storage capacity difference [{name_scen} – {name_base}] [Mton]",
        fontsize=14
    )
    ax.set_xlabel("Period", fontsize=14)
    ax.set_title(
        f"Hydrogen storage capacity per period\n{name_scen} – {name_base}",
        fontsize=16
    )

    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Storage type", fontsize=11, title_fontsize=12)

    plt.tight_layout()

    if savefigure and figurename and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        outpath = Path(results_dir) / f"{figurename}_H2storageCapDiff.png"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {outpath}")

    plt.show()

    return diff_Mton

df_store_flex   = pd.read_csv(result_dir2/"results_hydrogen_storage_inv.csv")
df_store_noflex = pd.read_csv(result_dir1/"results_hydrogen_storage_inv.csv")

diff_Mton = hydrogen_storage_capacity_diff(
    df_base=df_store_flex,
    df_scen=df_store_noflex,
    name_base="FLEX",
    name_scen="NOFLEX"
)














