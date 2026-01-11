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

result_dir= data_dir / "Results_FINAL_BASE_NOFLEX_emcap" / 'full_model_base'

result_dir2= data_dir / 'Results_FINAL_BASE_sys2' / 'full_model_base'
result_dir8=data_dir / 'Results_FINAL_BASE_sys8' / 'full_model_base'
result_dir16=data_dir / 'Results_FINAL_BASE_sys16' / 'full_model_base'
result_dir24=data_dir / 'Results_FINAL_BASE_sys24' / 'full_model_base'
result_dir32=data_dir / 'Results_FINAL_BASE_sys32' / 'full_model_base'

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
    line_color="gray",
    savefigure=False,
    figurename=None,
    results_dir=None,
):

    # ---------- Nodekoordinater ----------
    node_coords = {
        "Austria": (14.55, 47.59), "Belgium": (4.47, 50.85), "BosniaH": (17.67, 43.92),
        "Bulgaria": (25.48, 42.73), "Croatia": (15.98, 45.10), "CzechR": (15.47, 49.74),
        "Denmark": (10.0, 56.0), "France": (2.21, 46.22), "Germany": (10.45, 51.16),
        "GreatBrit.": (-2, 53), "Greece": (21.82, 39.07), "Hungary": (19.40, 47.16),
        "Italy": (12.57, 42.83), "Luxemb.": (6.13, 49.61), "Macedonia": (21.75, 41.61),
        "Netherlands": (5.29, 52.13), "NO1": (10.98, 60.62), "NO2": (7.38, 59.15),
        "NO3": (8.0, 62.47), "NO4": (19.0, 69.0), "NO5": (6.52, 60.57),
        "Poland": (19.14, 52.13), "Portugal": (-8.0, 39.5), "Romania": (24.96, 45.94),
        "Serbia": (20.45, 44.82), "Slovakia": (19.70, 48.66), "Slovenia": (14.51, 46.15),
        "Spain": (-3.7, 40.4), "Sweden": (15.00, 60.12), "Switzerland": (8.23, 46.80),
        "Ireland": (-8, 53.35), "Estonia": (25.0, 58.6), "Latvia": (24.1, 56.9),
        "Lithuania": (24.0, 55.3), "Finland": (25.0, 65.0),
    }

    # ---------- Oppsummering ----------
    demand_sectors = [
        'Power load [MWh]', 'Power for transport [MWh]', 'Power for steel [MWh]',
        'Power for cement [MWh]', 'Power for ammonia [MWh]',
        'Power reformer plant [MWh]', 'Power for NG [MWh]', 'Power for hydrogen [MWh]',
    ]

    cols_power = [
        "Node",
        "Power generation [MWh]",
        "Power transmission in [MWh]",
        "Power transmission out [MWh]",
    ] + demand_sectors

    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    grouped = (
        df_power[cols_power].groupby("Node").sum() * 5 * seasonScale / n_scen
    )
    load_shed = df_power.groupby("Node")["Power shed [MWh]"].sum() * 5 * seasonScale / n_scen

    grouped["Demand [MWh]"] = grouped[demand_sectors].sum(axis=1) - load_shed

    summary = grouped[[
        "Power generation [MWh]", "Power transmission in [MWh]",
        "Power transmission out [MWh]", "Demand [MWh]",
    ]]
    summary = summary.loc[summary.index.intersection(node_coords.keys())]
    summary["total"] = summary.sum(axis=1)

    # ---------- Transmission ----------
    df_trans = df_trans.copy()
    df_trans["node_pair"] = df_trans.apply(lambda r: tuple(sorted([r["FromNode"], r["ToNode"]])), axis=1)
    df_sum = df_trans.groupby("node_pair")["TransmissionReceived_MW"].sum().reset_index()
    df_sum = df_sum[df_sum["TransmissionReceived_MW"] > 1e-12]

    # ---------- Plot ----------
    fig = plt.figure(figsize=(14, 14))
    fig.subplots_adjust(bottom=0.25)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-11, 35, 35, 71])
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7)

    # =======================================================
    #   Transmission line thickness = sqrt-scaled
    # =======================================================
    legend_lines = []
    if not df_sum.empty:
        p98 = np.percentile(df_sum["TransmissionReceived_MW"], 98)
        eff_max = p98
        max_val = df_sum["TransmissionReceived_MW"].max()
        min_val = df_sum["TransmissionReceived_MW"].min()

        def scale_lw(v):
            v_clipped = min(v, eff_max)  # topper av
            v_norm = v_clipped / eff_max  # lineær skalering
            lw_min, lw_max = 1.2, 25
            return lw_min + v_norm * (lw_max - lw_min)

        for _, row in df_sum.iterrows():
            n1, n2 = row["node_pair"]
            if n1 in node_coords and n2 in node_coords:
                xs, ys = zip(node_coords[n1], node_coords[n2])
                ax.plot(xs, ys, color=line_color, linewidth=scale_lw(row["TransmissionReceived_MW"]),
                        alpha=0.45, transform=ccrs.PlateCarree(), zorder=1)

        def format_power(v):
            if v >= 1e6:
                return f"{v/1e6:.1f} TW"
            elif v >= 1e3:
                return f"{v/1e3:.1f} GW"
            return f"{v:.0f} MW"

        # Legend with representative levels
        levels = [
            min_val,
            np.percentile(df_sum["TransmissionReceived_MW"], 50),
            np.percentile(df_sum["TransmissionReceived_MW"], 90),
            eff_max
        ]

        legend_lines = [
            Line2D([], [], color=line_color,
                   linewidth=scale_lw(v),
                   label=format_power(v))
            for v in levels
        ]

    # =======================================================
    #   PIE PLOTTING
    # =======================================================
    colors_pie = ("#4C78A8", "#F58518", "#E45756", "#72B7B2")

    sizes = summary["total"].values
    tmax = sizes.max()
    r_min, r_max = 0.3, 2.6

    def scale_radius(v):
        return r_min + np.sqrt(v / tmax) * (r_max - r_min)

    def draw_pie(ax, lon, lat, values, r, colors):
        total = sum(values)
        if total <= 0:
            return
        fracs = np.array(values) / total
        start = 0
        for frac, col in zip(fracs, colors):
            if frac > 0:
                theta1, theta2 = 360 * start, 360 * (start + frac)
                wedge = Wedge((lon, lat), r, theta1, theta2, facecolor=col,
                              edgecolor="black", linewidth=0.3,
                              transform=ccrs.PlateCarree(), zorder=3)
                ax.add_patch(wedge)
                start += frac
        ring = Circle((lon, lat), r, facecolor="none",
                      edgecolor="black", linewidth=0.4,
                      transform=ccrs.PlateCarree(), zorder=4)
        ax.add_patch(ring)

    for node, row in summary.iterrows():
        lon, lat = node_coords[node]
        r = scale_radius(row["total"])
        vals = [
            row["Power generation [MWh]"],
            row["Power transmission in [MWh]"],
            row["Power transmission out [MWh]"],
            row["Demand [MWh]"],
        ]
        draw_pie(ax, lon, lat, vals, r, colors_pie)

    # =======================================================
    #   PIE SIZE LEGEND (Alternativ A)
    # =======================================================
    pie_totals = summary["total"].values
    pie_levels = np.linspace(pie_totals.min(), pie_totals.max(), 4)

    pie_size_legend = [
        Line2D([], [], marker='o', linestyle='None',
               markersize=scale_radius(val) * 10,
               markerfacecolor="lightgray", markeredgecolor="black",
               label=f"{val:,.0f} MWh")
        for val in pie_levels
    ]

    # =======================================================
    #   PIE COLOR LEGEND
    # =======================================================
    pie_color_legend = [
        Line2D([], [], marker='o', color="w", label="Local generation",
               markerfacecolor=colors_pie[0], markersize=12),
        Line2D([], [], marker='o', color="w", label="Import",
               markerfacecolor=colors_pie[1], markersize=12),
        Line2D([], [], marker='o', color="w", label="Export",
               markerfacecolor=colors_pie[2], markersize=12),
        Line2D([], [], marker='o', color="w", label="Demand",
               markerfacecolor=colors_pie[3], markersize=12),
    ]

    # =======================================================
    #   PLACE LEGENDS IN ORDER
    # =======================================================
    # Pies: fargelegend
    leg1 = ax.legend(handles=pie_color_legend, loc="lower left",
                     frameon=True, fontsize=11,
                     bbox_to_anchor=(0.13, -0.18),
                     labelspacing=1.4,  # ekstra luft for store sirkler
                     handletextpad=1.0,  # større avstand mellom sirkel og tekst
                     borderpad=0.7,
                     )

    ax.add_artist(leg1)

    # Pie size-legend
    leg2 = ax.legend(handles=pie_size_legend, loc="lower left",
                     bbox_to_anchor=(0.45, -0.18),
                     frameon=True, fontsize=11,
                     title="Energy content", title_fontsize=12,
                     labelspacing=1.4,  # ekstra luft for store sirkler
                     handletextpad=1.0,  # større avstand mellom sirkel og tekst
                     borderpad=0.7,
                     )
    ax.add_artist(leg2)

    # Transmission legend
    ax.legend(handles=legend_lines, loc="lower left",
              bbox_to_anchor=(0.77, -0.18),
              frameon=True, fontsize=11,
              title="Transmission flow", title_fontsize=12,
              labelspacing=1.4,  # ekstra luft for store sirkler
              handletextpad=1.5,  # større avstand mellom sirkel og tekst
              borderpad=0.7,
              )

    plt.tight_layout()
    plt.show()


def system_costs_analysis(result_dir):
    df = pd.read_csv((result_dir / "results_output_EuropeSummary.csv"), delimiter=",", skiprows=16,
                                 usecols=[0, 1, 2, 3, 4, 5], skipfooter=16, engine='python')

    df["genProduction_5yr_GWh"] = df["genExpectedAnnualProduction_GWh"] * 5 * 0.001

    # Gjør GeneratorType lowercase for robust matching
    df["GeneratorType_lower"] = df["GeneratorType"].str.lower()

    # Definer teknologinøkler
    nuclear_keys = ["nuclear"]
    windon_keys = ["windonshore"]
    solar_keys = ["solar"]

    # Summer produksjon (5-årsperioder)
    nuc = df.loc[
        df["GeneratorType_lower"].str.contains("|".join(nuclear_keys)),
        "genProduction_5yr_GWh"
    ].sum()

    windon = df.loc[
        df["GeneratorType_lower"].str.contains("|".join(windon_keys)),
        "genProduction_5yr_GWh"
    ].sum()

    solar = df.loc[
        df["GeneratorType_lower"].str.contains("|".join(solar_keys)),
        "genProduction_5yr_GWh"
    ].sum()

    # Other = alt som ikke er nuclear, windonshore eller solar
    mask_other = ~(
        df["GeneratorType_lower"].str.contains("|".join(nuclear_keys + windon_keys + solar_keys))
    )

    other = df.loc[mask_other, "genProduction_5yr_GWh"].sum()

    # Resultat
    print(f"nuc    = {nuc:.2f} GWh (2020–2055)")
    print(f"windon = {windon:.2f} GWh (2020–2055)")
    print(f"solar  = {solar:.2f} GWh (2020–2055)")
    print(f"other  = {other:.2f} GWh (2020–2055)")

    return nuc, windon, solar, other


nuc2, windon2, solar2, other2 =system_costs_analysis(result_dir2)
nuc8,windon8,solar8,other8 = system_costs_analysis(result_dir8)
nuc16,windon16,solar16,other16 = system_costs_analysis(result_dir16)
nuc24,windon24,solar24,other24 = system_costs_analysis(result_dir24)
nuc32,windon32,solar32,other32 = system_costs_analysis(result_dir32)

sys_costs = np.array([2, 8, 16, 24, 32], dtype=float)

nuc    = np.array([nuc2,  nuc8,  nuc16,  nuc24,  nuc32], dtype=float)
windon = np.array([windon2, windon8, windon16, windon24, windon32], dtype=float)
solar  = np.array([solar2, solar8, solar16, solar24, solar32], dtype=float)
other  = np.array([other2, other8, other16, other24, other32], dtype=float)

import numpy as np

# Finn intervallet der kurvene krysser
for i in range(len(sys_costs) - 1):
    if (nuc[i] - windon[i]) * (nuc[i+1] - windon[i+1]) < 0:
        idx = i
        break

# Punktene
x1, x2 = sys_costs[idx], sys_costs[idx+1]
n1, n2 = nuc[idx], nuc[idx+1]
w1, w2 = windon[idx], windon[idx+1]

# Lineær funksjon: y = a x + b
a_n = (n2 - n1) / (x2 - x1)
b_n = n1 - a_n * x1

a_w = (w2 - w1) / (x2 - x1)
b_w = w1 - a_w * x1

# Skjæringspunkt
break_even_cost = (b_w - b_n) / (a_n - a_w)
break_even_prod = a_n * break_even_cost + b_n

print(f"Break-even ≈ {break_even_cost:.2f} EUR/MWh")
print(f"Production at break-even ≈ {break_even_prod:.1f} TWh")


plt.figure(figsize=(10,7))

techs = {
    "Nuclear": nuc,
    "Wind onshore": windon,
    "Solar": solar,
    "Other": other
}

colors=['pink','seagreen','violet','darkgray']

for (name, values), color in zip(techs.items(), colors):
    plt.plot(
        sys_costs,
        values,
        marker="o",
        linewidth=2.5,
        label=name,
        color=color
    )

    # legg til tall ved hvert punkt
    for x, y in zip(sys_costs, values):
        plt.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(3, 6),
            ha="left",
            fontsize=12,
            color='black'
        )

plt.scatter(
    break_even_cost,
    break_even_prod,
    color="red",
    s=140,
    zorder=6,
    label=f"{break_even_cost:.2f} EUR/MWh",
)


plt.legend(fontsize=12)


plt.xlabel("Balance costs VRES [EUR/MWh] ", fontsize=16)
plt.ylabel("Total energy production [TWh]", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0,11*10e3)
plt.legend(fontsize=12)
plt.grid(linestyle='--')
plt.tight_layout()
plt.show()



trans_map='no'

if trans_map == 'yes':

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






