from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import geopandas as gpd
from shapely.geometry import LineString
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Wedge, Circle

project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"

result_dir= data_dir / 'Results_BASE_new' / 'full_model_base'

"""
result_dir1 = data_dir / "Results_BalanceC8_trans" / "full_model_base"
result_dir2 = data_dir / 'Results_BalanceCosts_trans' / 'full_model_base'
result_dir3 = data_dir / 'Results_balanceC55_trans' / 'full_model_base'
result_dir4 = data_dir / 'Results_balanceC2_5' / 'full_model_base'

# In[] Fornybar energi (variabel) produksjon VS balansekostnader

VRES=['Windonshore']

Elec_gentype1=pd.read_csv(result_dir1 / 'results_elec_generation_inv.csv')
Elec_gentype2=pd.read_csv(result_dir2 / 'results_elec_generation_inv.csv')
Elec_gentype3=pd.read_csv(result_dir3 / 'results_elec_generation_inv.csv')
Elec_gentype4 = pd.read_csv(result_dir4 / 'results_elec_generation_inv.csv')
def Balance_analysis(Elec_gentype,VRES):
    TotalGen_perTech = (
            Elec_gentype
            .groupby(['GeneratorType'], as_index=False)['genExpectedAnnualProduction_GWh']
            .sum()
        )
    TotalGen_perTech['genExpectedAnnualProduction_GWh']=TotalGen_perTech['genExpectedAnnualProduction_GWh']*5/1000


    VRES_prod=0 #GWh
    Nuc_prod=TotalGen_perTech.loc[TotalGen_perTech['GeneratorType']=='Nuclear','genExpectedAnnualProduction_GWh'].iloc[0]
    Other=0

    for i in VRES:
        VRES_prod+=TotalGen_perTech.loc[TotalGen_perTech['GeneratorType']==i,'genExpectedAnnualProduction_GWh'].iloc[0]

    for i in TotalGen_perTech['GeneratorType']:
        Other+=TotalGen_perTech.loc[TotalGen_perTech['GeneratorType']==i,'genExpectedAnnualProduction_GWh'].iloc[0]
    Other=Other-VRES_prod-Nuc_prod

    return VRES_prod,Nuc_prod,Other

VRES1,Nuc1,Other1=Balance_analysis(Elec_gentype1,VRES)
VRES2,Nuc2,Other2=Balance_analysis(Elec_gentype2,VRES)
VRES3,Nuc3,Other3=Balance_analysis(Elec_gentype3,VRES)
VRES4,Nuc4,Other4 = Balance_analysis(Elec_gentype4,VRES)

# Data
x = np.array([2.55,4.67, 8.49, 16.98])
VRES_list = np.array([VRES4,VRES3, VRES1, VRES2])
Nuc_list  = np.array([Nuc4,Nuc3,  Nuc1,  Nuc2])
Other_list= np.array([Other4,Other3,Other1,Other2])

# Lineær fit (y = a x + b)
a_vres,  b_vres  = np.polyfit(x, VRES_list, 1)
a_nuc,   b_nuc   = np.polyfit(x, Nuc_list,  1)
a_other, b_other = np.polyfit(x, Other_list,1)

# Tett x-akse for glatte linjer
xfit = np.linspace(x.min(), x.max(), 200)

# Beregn linjene
yfit_vres  = a_vres*xfit  + b_vres
yfit_nuc   = a_nuc*xfit   + b_nuc
yfit_other = a_other*xfit + b_other

# Plot datapunkter
plt.figure(figsize=(10,6))
plt.plot(x, VRES_list,  marker='o',markersize='8', color='thistle',  label=f'VRES',linewidth=3)
plt.plot(x, Nuc_list,    marker='o',markersize='8', color='steelblue',   label=f'Nuclear',linewidth=3)
plt.plot(x, Other_list,  marker='o',markersize='8', color='darkseagreen', label=f'Other',linewidth=3)

# Plot regresjonslinjer
plt.plot(xfit, yfit_vres,  color='thistle',linestyle='--',  label=f'a={a_vres:.3f}')
plt.plot(xfit, yfit_nuc,   color='steelblue',linestyle='--',   label=f'a={a_nuc:.3f}')
plt.plot(xfit, yfit_other, color='darkseagreen',linestyle='--', label=f'a={a_other:.3f}')

plt.xlabel('Balance and reserve costs for VRES [EUR/MWh]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Total power production [TWh]',fontsize=14)
plt.ylim(50000,92000)
plt.grid(True)
plt.legend(loc='upper right',title='Energy source', fontsize=12,title_fontsize=14)
plt.tight_layout()
plt.savefig(data_dir/'Balance_costs_impact.eps', format='eps', bbox_inches='tight')
plt.show()
"""
from functions_plot import plot_power_balance_for_high_h2, plot_power_demand,P_prodImportExportDemand_piechart

Power_balance=pd.read_csv(result_dir/'results_power_balance.csv')
plot_power_demand(Power_balance,n_hours=12)
plot_power_balance_for_high_h2(Power_balance,'2050-2055','scenario1','Italy')

P_prodImportExportDemand_piechart(Power_balance, 12, 2, savefigure=False, figurename=None, results_dir=None)

Transmission_operation=pd.read_csv(result_dir/'results_elec_transmission_operational.csv')

def power_pies_and_transmission_map(
    df_power,
    df_trans,
    n_hours,
    n_scen,
    line_color="tab:gray",
    savefigure=False,
    figurename=None,
    results_dir=None,
):
    # ---------- 1) Felles node-koordinater ----------
    node_coords = {
        "Austria": (14.55, 47.59),
        "Belgium": (4.47, 50.85),
        "BosniaH": (17.67, 43.92),
        "Bulgaria": (25.48, 42.73),
        "Croatia": (15.98, 45.10),
        "CzechR": (15.47, 49.74),
        "Denmark": (10.0, 56.0),
        "France": (2.21, 46.22),
        "Germany": (10.45, 51.16),
        "GreatBrit.": (-2, 53),
        "Greece": (21.82, 39.07),
        "Hungary": (19.40, 47.16),
        "Italy": (12.57, 42.83),
        "Luxemb.": (6.13, 49.61),
        "Macedonia": (21.75, 41.61),
        "Netherlands": (5.29, 52.13),
        "NO1": (10.98, 60.62),
        "NO2": (7.38, 59.15),
        "NO3": (8.0, 62.47),
        "NO4": (19.0, 69.0),
        "NO5": (6.52, 60.57),
        "Poland": (19.14, 52.13),
        "Portugal": (-8.0, 39.5),
        "Romania": (24.96, 45.94),
        "Serbia": (20.45, 44.82),
        "Slovakia": (19.70, 48.66),
        "Slovenia": (14.51, 46.15),
        "Spain": (-3.7, 40.4),
        "Sweden": (15.00, 60.12),
        "Switzerland": (8.23, 46.80),
        "Ireland": (-8, 53.35),
        "Estonia": (25.0, 58.6),
        "Latvia": (24.1, 56.9),
        "Lithuania": (24.0, 55.3),
        "Finland": (25.0, 61.0),
    }

    # ======================================================================
    # 2) PIES fra df_power: prod / import / export / demand
    # ======================================================================
    demand_sectors = [
        'Power load [MWh]',
        'Power for transport [MWh]',
        'Power for steel [MWh]',
        'Power for cement [MWh]',
        'Power for ammonia [MWh]',
        'Power reformer plant [MWh]',
        'Power for NG [MWh]',
        'Power for hydrogen [MWh]',

    ]

    cols_power = [
        "Node",
        "Power generation [MWh]",
        "Power transmission in [MWh]",
        "Power transmission out [MWh]",
    ] + demand_sectors

    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    grouped = (
        df_power[cols_power]
        .groupby("Node")
        .sum()
        * 5 * seasonScale / n_scen
    )
    load_shed = (
            df_power.groupby("Node")["Power shed [MWh]"].sum() * 5 * seasonScale / n_scen
    )

    # ---- Demand = sektorbruk minus load shed ----
    grouped["Demand [MWh]"] = grouped[demand_sectors].sum(axis=1) - load_shed

    summary = grouped[[
        "Power generation [MWh]",
        "Power transmission in [MWh]",
        "Power transmission out [MWh]",
        "Demand [MWh]",
    ]].copy()

    summary = summary.loc[summary.index.intersection(node_coords.keys())].copy()
    summary["total"] = summary.sum(axis=1)

    # ======================================================================
    # 3) TRANSMISJON fra df_trans (FromNode / ToNode / TransmissionReceived_MW)
    # ======================================================================
    df_trans = df_trans.copy()

    # gjør forbindelser retningsuavhengige
    df_trans["node_pair"] = df_trans.apply(
        lambda row: tuple(sorted([row["FromNode"], row["ToNode"]])),
        axis=1
    )

    # summer over timer/scenarier/perioder
    df_sum = (
        df_trans.groupby("node_pair")["TransmissionReceived_MW"]
        .sum()
        .reset_index()
    )

    # dropp “null-linjer”
    eps = 1e-12
    df_sum = df_sum[df_sum["TransmissionReceived_MW"] > eps]

    lines = []
    for _, row in df_sum.iterrows():
        n1, n2 = row["node_pair"]
        coord1 = node_coords.get(n1)
        coord2 = node_coords.get(n2)
        if coord1 and coord2:
            lines.append({
                "coords": [coord1, coord2],
                "value": row["TransmissionReceived_MW"],
                "nodes": f"{n1}–{n2}",
            })

    # ======================================================================
    # 4) Kart, linjer og pies på samme aksen
    # ======================================================================
    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-11, 35, 35, 71], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7)

    # ---- transmisjonslinjer med diskret tykkelse ----
    legend_lines = []
    if not df_sum.empty:
        max_val = max(df_sum["TransmissionReceived_MW"])
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        widths = [2, 30, 40, 50]

        import matplotlib.lines as mlines

        for l in lines:
            xs, ys = zip(*l["coords"])
            ratio = l["value"] / max_val
            for i in range(len(bins) - 1):
                if bins[i] <= ratio < bins[i + 1]:
                    lw = widths[i]
                    break
            else:
                lw = widths[-1]

            ax.plot(xs, ys, color=line_color, linewidth=lw, alpha=0.5,
                    transform=ccrs.PlateCarree())

        # linje-legend
        for i in range(len(widths)):
            lower_val = bins[i] * max_val
            if lower_val == 0:
                lower_val = df_sum.loc[df_sum["TransmissionReceived_MW"] > 0,
                                       "TransmissionReceived_MW"].min()
            upper_val = bins[i + 1] * max_val
            label = f"{lower_val:.1f} – {upper_val:.1f} MW"
            legend_lines.append(
                mlines.Line2D([], [], color=line_color,
                              linewidth=widths[i], label=label)
            )

    # ---- hjelpefunksjon for pies ----
    def draw_pie(ax, lon, lat, values, radius_deg, colors):
        total = float(np.sum(values))
        if total <= 0:
            return
        fracs = np.array(values) / total
        start = 0.0
        for frac, col in zip(fracs, colors):
            if frac <= 0:
                continue
            theta1, theta2 = 360 * start, 360 * (start + frac)
            wedge = Wedge(
                (lon, lat), radius_deg, theta1, theta2,
                facecolor=col, edgecolor="black", linewidth=0.3,
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(wedge)
            start += frac
        ring = Circle(
            (lon, lat), radius_deg, facecolor="none",
            edgecolor="black", linewidth=0.3,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(ring)

    # skalering for piestørrelse
    t = summary["total"].values
    if len(t) == 0:
        tmax = 0
    else:
        tmax = np.nanmax(t)
    r_min, r_max = 0.3, 2.5

    def scale_radius(total):
        if tmax <= 0:
            return (r_min + r_max) / 2
        s = np.sqrt(total / tmax)
        return r_min + s * (r_max - r_min)

    # farger: prod, import, export, demand
    colors_pie = ("#4C78A8", "#F58518", "#E45756", "#72B7B2")

    # tegn pies
    for node, row in summary.iterrows():
        lon, lat = node_coords[node]
        r = scale_radius(row["total"])
        vals = [
            row["Power generation [MWh]"],
            row["Power transmission in [MWh]"],
            row["Power transmission out [MWh]"],
            row["Demand [MWh]"],
        ]
        draw_pie(ax, lon, lat, vals, radius_deg=r, colors=colors_pie)

    # ---- samlet legend (pies + linjer) ----
    pie_legend = [
        Line2D([0], [0], marker="o", color="w", label="Local generation",
               markerfacecolor=colors_pie[0], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Import",
               markerfacecolor=colors_pie[1], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Export",
               markerfacecolor=colors_pie[2], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Demand (all sectors)",
               markerfacecolor=colors_pie[3], markersize=12),
    ]

    handles = pie_legend + legend_lines
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        fontsize=11,
        title="Pies: energy shares  |  Lines: avg flow",
        title_fontsize=12,
    )

    plt.tight_layout()

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_power_pies_plus_trans.png"
        plt.savefig(figpath, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {figpath}")

    plt.show()

power_pies_and_transmission_map(
    Power_balance,
    Transmission_operation,
    12,
    2,
    line_color="tab:gray",
    savefigure=False,
    figurename=None,
    results_dir=None,
)






