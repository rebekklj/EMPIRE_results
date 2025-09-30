import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import geopandas as gpd
from shapely.geometry import LineString
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Wedge, Circle

# ------------------PLOTTING AV INSTALLERT KAPASITET PER STØRSTE TEKNOLOGIER--------------------------------

def Plot_Installed_capacity_per_tech_split(df,
                                           threshold=90_000,
                                           figsize=(16, 12),
                                           savefigure=False,
                                           figurename=None,
                                           results_dir=None):
    # 1) Data preparation
    work = df.copy()
    work[['genInvCap_MW', 'genInstalledCap_MW']] = work[
        ['genInvCap_MW', 'genInstalledCap_MW']
    ].apply(pd.to_numeric, errors='coerce')

    # Existing capacity at start of each period
    work['existing_MW'] = (work['genInstalledCap_MW'] - work['genInvCap_MW']).clip(lower=0)

    # Aggregate per generator type & period
    new_cap = work.pivot_table(
        index='GeneratorType', columns='Period',
        values='genInvCap_MW', aggfunc='sum', fill_value=0
    )
    old_cap = work.pivot_table(
        index='GeneratorType', columns='Period',
        values='existing_MW', aggfunc='sum', fill_value=0
    )

    # Sort periods chronologically by extracting start year
    def get_start_year(period_str):
        m = re.match(r'(\d{4})', period_str)
        return int(m.group(1)) if m else float('inf')

    periods = sorted(new_cap.columns, key=get_start_year)
    new_cap = new_cap[periods]
    old_cap = old_cap[periods]

    # Filter out small technologies
    keep = (new_cap + old_cap).abs().max(axis=1) >= threshold
    new_cap = new_cap.loc[keep]
    old_cap = old_cap.loc[keep]

    # 2) Color setup
    base_colors = ['lightskyblue', 'teal', 'darkseagreen',
                   'khaki', 'plum', 'darkslateblue', 'lightsteelblue']

    # 3) Plot
    ntech = len(new_cap)
    barw = 0.8 / len(periods)
    indices = np.arange(ntech)

    fig, ax = plt.subplots(figsize=figsize)

    sum_ins = 0
    for j, p in enumerate(periods):
        color_new = base_colors[j]
        color_old = to_rgba(color_new, alpha=0.4)

        # legacy (existing) capacity
        ax.bar(indices + j * barw,
               old_cap[p] / 1000000,
               barw,
               color=color_old,
               edgecolor='none')

        # new builds
        ax.bar(indices + j * barw,
               new_cap[p] / 1000000,
               barw,
               bottom=old_cap[p] / 1000000,
               color=color_new,
               edgecolor='black',
               linewidth=0.3,
               label=p)
        if p == '2050-2055':
            sum_ins += sum(new_cap[p]) + sum(old_cap[p])

    print(sum_ins)

    ax.set_axisbelow(True)
    ax.set_ylabel('Installed capacity [TW]', fontsize=26)
    ax.set_xlabel('Technology', fontsize=26)
    ax.set_xticks(indices + barw * (len(periods) - 1) / 2)
    ax.set_ylim(0, 4)
    ax.set_xticklabels(new_cap.index, rotation=25, ha='right', fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Legend outside
    ax.legend(title='Period', loc='upper right', title_fontsize=26, fontsize=24)

    # 4) Inset for Nuclear
    labels = [t.get_text() for t in ax.get_xticklabels()]
    if 'Nuclear' in labels:
        nuc_idx = labels.index('Nuclear')
        nuc_x = ax.get_xticks()[nuc_idx]
        group_w = barw * len(periods)
        ymax = (max(rect.get_height()
                   for cont in ax.containers
                   for rect in cont
                   if nuc_x - group_w / 2 <= rect.get_x() <= nuc_x + group_w / 2) + 0.01)

        axins = inset_axes(ax, width='30%', height='45%',
                           loc='upper left',
                           bbox_to_anchor=(0.1, 0.01, 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=1.2)
        for cont in ax.containers:
            for rect in cont:
                if nuc_x - group_w / 2 <= rect.get_x() <= nuc_x + group_w / 2:
                    axins.bar(rect.get_x(), rect.get_height(),
                              rect.get_width(), bottom=rect.get_y(),
                              color=rect.get_facecolor(),
                              alpha=rect.get_alpha())
        axins.set_xlim(nuc_x - group_w / 2, nuc_x + group_w / 2)
        axins.set_ylim(0, ymax * 1.1)
        axins.set_xticks([])
        axins.tick_params(axis='y', labelsize=24)
        axins.set_ylabel('TW', fontsize=26)
        mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='gray')

    plt.tight_layout()

    if savefigure and figurename and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f'{figurename}_insCap.png'
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {figpath}')

    plt.show()


def Expected_annual_production(Elec_generation_countries, savefigure=False, results_dir=None, figurename=None):
    # ---- Palett (din) ----
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
        'Liginiteexisting': 'maroon',  # merk stavemåten, behold den om det er slik i dataene
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
    fallback_color = 'lightgrey'

    # ---- Aggreger ----
    cols = ['GeneratorType', 'Period', 'genExpectedAnnualProduction_GWh']
    Elec_gentype = Elec_generation_countries[cols].copy()

    AnnualGen_perTech = (
        Elec_gentype
        .groupby(['GeneratorType', 'Period'], as_index=False)['genExpectedAnnualProduction_GWh']
        .sum()
    )

    # Pivot: rader=GeneratorType, kolonner=Period
    df_pivot = (
        AnnualGen_perTech
        .pivot(index='GeneratorType', columns='Period', values='genExpectedAnnualProduction_GWh')
        .fillna(0.0)
    )

    # Fjern teknologier med bare nuller
    df_pivot = df_pivot.loc[df_pivot.sum(axis=1) != 0]

    # Sortér perioder og skaler til TWh
    df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1) / 1000.0

    # ---- Bygg fargeliste i riktig rekkefølge ----
    tech_order = df_pivot.index.tolist()
    color_list = [tech_colors.get(t, fallback_color) for t in tech_order]

    # (valgfritt) logg teknologier uten definert farge
    missing = [t for t, c in zip(tech_order, color_list) if c == fallback_color and t not in tech_colors]
    if missing:
        print("Obs: mangler farge for teknologier ->", ", ".join(missing))

    # ---- Plot: stacked area (perioder på x-aksen) ----
    # .T fordi vi vil ha Period på x-aksen og teknologier stablet
    ax = df_pivot.T.plot(
        kind='area',
        stacked=True,
        figsize=(14, 10),
        color=color_list,
        alpha=0.7
    )

    ax.set_axisbelow(True)
    ax.set_xlabel("Investment period", fontsize=20)
    ax.set_ylabel("Expected annual production (TWh)", fontsize=20)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True, axis='y')

    # Legende: behold rekkefølge som i data (samme som fargerekkefølgen)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, labels=tech_order,
        title="Generator Type", title_fontsize=16,
        loc="upper center", bbox_to_anchor=(0.5, -0.1),
        ncol=3, fontsize=16
    )

    plt.tight_layout()

    # Lagring
    if savefigure and figurename and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f'{figurename}_AnnELprod.png'
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {figpath}')

    plt.show()


def Yearly_hydrogenProd_perTech(
        df, x1, x2, x3, x4, n_scen, n_hours,
        y_max=None, savefigure=False, results_dir=None, figurename=None
):
    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)
    # 1) Aggreger dataene per periode
    agg = (df.groupby("Period")
           .agg({x1: "sum", x2: "sum", x3: "sum", x4: "sum"})
           .reset_index())

    # 2) Sørg for korrekt rekkefølge på periodene
    periods = ["2020-2025", "2025-2030", "2030-2035", "2035-2040",
               "2040-2045", "2045-2050", "2050-2055"]
    agg["Period"] = pd.Categorical(agg["Period"],
                                   categories=periods,
                                   ordered=True)
    agg = agg.sort_values("Period")

    # 3) Plot som stablede stolper
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = np.arange(len(agg)) * 0.5  # x-posisjoner for stolpene
    width = 0.3

    agg[[x1, x2, x3, x4]] = agg[[x1, x2, x3, x4]] * seasonScale / n_scen

    # Beregn totalproduksjon per periode for prosentandel
    total = agg[[x1, x2, x3, x4]].sum(axis=1)

    # Stablet: bygg “bottom” fortløpende
    bottom = np.zeros(len(agg))
    for col, label, color in zip(
            [x1, x2, x3, x4],
            ["PEM", "Alkaline", "SOEC", "Reformer"],
            ["steelblue", "plum", "rebeccapurple", "grey"]):

        values = agg[col] / 1e6
        bars = ax.bar(pos, values, width, bottom=bottom, label=label, color=color, alpha=0.8)

        # Legg til prosenttekst i hver del
        for i in range(len(agg)):
            percent = (agg[col][i] / total[i]) * 100 if total[i] > 0 else 0
            if percent > 3:  # vis kun dersom >3% for å unngå rot
                ax.text(pos[i], bottom[i] + values[i] / 2,
                        f"{percent:.0f}%", ha='center', va='center', fontsize=14, color='white')

        bottom += values

    # 4) Akseoppsett
    ax.set_xticks(pos)
    ax.set_xticklabels(agg["Period"])
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("Investment period", fontsize=18)
    ax.set_ylabel("Annual hydrogen production [M ton]", fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=16, title='Technology', title_fontsize=18)

    if y_max is not None:
        ax.set_ylim(0, y_max)
    if savefigure and figurename and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f'{figurename}_AnnH2prod.png'
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {figpath}')
    plt.show()


def plot_h2_demand_stacked_area(
        df,
        sectors=[
            'Hydrogen used for transport [ton]',
            'Hydrogen burned for power and heat [ton]',
            'Hydrogen used for steel [ton]',
            'Hydrogen used for cement [ton]',
            'Hydrogen used for ammonia [ton]',
            'Hydrogen used for oil refining [ton]'],

        period_order=("2020-2025", "2025-2030", "2030-2035", "2035-2040",
                      "2040-2045", "2045-2050", "2050-2055"),
        n_hours=None,
        y_max=None,
        savefigure=False,
        results_dir=None,
        figurename=None
):
    # Sjekk at alle kolonner finnes
    missing = [c for c in sectors + ['Period', 'Scenario'] if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i df: {missing}")

    # 1) Aggreger innen hver (Period, Scenario) over alle andre dimensjoner
    #    (Node, Season, Hour, osv. blir summer)
    per_per_sc = (df.groupby(['Period', 'Scenario'])[sectors]
                  .sum()
                  .reset_index())

    # 2) Ta gjennomsnitt over scenarier (like sannsynligheter)
    #    NB: Dette tilsvarer å dele på antall scenarier per periode.
    agg = (per_per_sc.groupby('Period')[sectors]
           .mean()
           .reset_index())

    # 3) Sett korrekt rekkefølge på periodene
    agg['Period'] = pd.Categorical(agg['Period'], categories=period_order, ordered=True)
    agg = agg.sort_values('Period')

    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    # 4) Konverter til Mton og skaler for timer
    agg_mton = agg.copy()
    agg_mton[sectors] = agg_mton[sectors] * seasonScale / 1e6

    # 5) Plot: stacked area per periode
    x = np.arange(len(agg_mton['Period']))
    y = [agg_mton[c].values for c in sectors]

    colors = ['darkslategrey', 'mediumturquoise', 'paleturquoise', 'gold', 'orange', 'seagreen']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(x, y, labels=[s.replace('Hydrogen ', '').replace(' [ton]', '') for s in sectors], colors=colors,
                 alpha=0.5)

    # Akser og pynt
    ax.set_xticks(x)
    ax.set_xticklabels(agg_mton['Period'])
    ax.set_xlabel("Investment period", fontsize=14)
    ax.set_ylabel("Hydrogen demand [M ton/yr]", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", ncol=2, fontsize=11)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Valgfritt: totaletiketter på toppen av hvert punkt
    totals = agg_mton[sectors].sum(axis=1).values
    for i, t in enumerate(totals):
        ax.text(i, t, f"{t:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_H2_demands.png"
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {figpath}")

    plt.show()

def plot_power_demand(
        df,
        sectors=['Power load [MWh]','Power for transport [MWh]','Power for steel [MWh]','Power for cement [MWh]','Power for ammonia [MWh]','Power reformer plant [MWh]','Power for NG [MWh]','Power shed [MWh]','Power for hydrogen [MWh]'],

        period_order=("2020-2025", "2025-2030", "2030-2035", "2035-2040",
                      "2040-2045", "2045-2050", "2050-2055"),
        n_hours=None,
        y_max=None,
        savefigure=False,
        results_dir=None,
        figurename=None
):
    # Sjekk at alle kolonner finnes
    missing = [c for c in sectors + ['Period', 'Scenario'] if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i df: {missing}")

    # 1) Aggreger innen hver (Period, Scenario) over alle andre dimensjoner
    #    (Node, Season, Hour, osv. blir summer)
    per_per_sc = (df.groupby(['Period', 'Scenario'])[sectors]
                  .sum()
                  .reset_index())

    # 2) Ta gjennomsnitt over scenarier (like sannsynligheter)
    #    NB: Dette tilsvarer å dele på antall scenarier per periode.
    agg = (per_per_sc.groupby('Period')[sectors]
           .mean()
           .reset_index())

    # 3) Sett korrekt rekkefølge på periodene
    agg['Period'] = pd.Categorical(agg['Period'], categories=period_order, ordered=True)
    agg = agg.sort_values('Period')

    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    # 4) Konverter til Mton og skaler for timer
    agg_mton = agg.copy()
    agg_mton[sectors] = agg_mton[sectors] * seasonScale / 1e6

    # 5) Plot: stacked area per periode
    x = np.arange(len(agg_mton['Period']))
    y = [agg_mton[c].values for c in sectors]

    colors = ['darkslategrey','teal','paleturquoise', 'gold', 'orange', 'seagreen','hotpink','lime','rebeccapurple']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(x, y, labels=[s.replace('Power ', '').replace(' [MWh]', '') for s in sectors], colors=colors,
                 alpha=0.5)

    # Akser og pynt
    ax.set_xticks(x)
    ax.set_xticklabels(agg_mton['Period'])
    ax.set_xlabel("Investment period", fontsize=14)
    ax.set_ylabel("Power demand [TWh/yr]", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", ncol=2, fontsize=11)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Valgfritt: totaletiketter på toppen av hvert punkt
    totals = agg_mton[sectors].sum(axis=1).values
    for i, t in enumerate(totals):
        ax.text(i, t, f"{t:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_EL_demands.png"
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {figpath}")

    plt.show()


def transmission_map(df, savefigure=False, figurename=None, results_dir=None):
    # === 2. Lag retningsuavhengige nodepar ===
    df["node_pair"] = df.apply(lambda row: tuple(sorted([row["FromNode"], row["ToNode"]])), axis=1)

    # === 3. Summer total flyt for hver nodekombinasjon og del på 2 (scenario-vektet) ===
    df_sum = (
        df.groupby("node_pair")["TransmissionReceived_MW"]
        .sum()
        .div(2)  # for 2 like vektede scenarier
        .reset_index()
    )

    # === 4. Definer (forenklede) koordinater for nodene ===
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
        "Finland": (25.0, 61.0)
    }

    lines = []
    for _, row in df_sum.iterrows():
        node1, node2 = row["node_pair"]
        coord1 = node_coords.get(node1)
        coord2 = node_coords.get(node2)
        if coord1 and coord2:
            lines.append({
                "coords": [coord1, coord2],
                "value": row["TransmissionReceived_MW"],
                "nodes": f"{node1}–{node2}"
            })

    # === 6. Plot med cartopy ===
    import numpy as np
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-9, 30, 34, 70], crs=ccrs.PlateCarree())

    # Bakgrunnskart
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    # ax.gridlines(draw_labels=False)

    # === Diskretisering i 5 tykkelsesnivåer ===
    max_val = 16879307.31535903
    bins = [0, 0.25, 0.5, 0.75, 1.0]  # proporsjoner av max
    widths = [1, 7, 14, 21]  # tykkelser du ønsker
    labels = []

    # Tegn kraftlinjer med diskrete nivåer
    for l in lines:
        xs, ys = zip(*l["coords"])
        ratio = l["value"] / max_val
        for i in range(len(bins) - 1):
            if bins[i] <= ratio < bins[i + 1]:
                lw = widths[i]
                break
        else:
            lw = widths[-1]  # fallback
        l["linewidth"] = lw
        ax.plot(xs, ys, color='firebrick', linewidth=lw, alpha=0.5, transform=ccrs.PlateCarree())

    # === Forklaringsboks for tykkelse → kraftmengde ===
    legend_lines = []
    for i in range(len(widths)):
        val = (bins[i + 1]) * max_val
        label = f"{val / 1000:.0f} GW"  # konverter fra MW til GW
        legend_lines.append(
            mlines.Line2D([], [], color='firebrick', linewidth=widths[i], label=label)
        )

    ax.legend(
        handles=legend_lines,
        title="Transmission flow",
        loc="upper left",
        frameon=True,
        labelspacing=1.5,
        fontsize=16,
        title_fontsize=18
    )

    # Tegn nodene
    for name, (lon, lat) in node_coords.items():
        ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
        ax.text(lon + 0.3, lat + 0.2, name, fontsize=10, transform=ccrs.PlateCarree())

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_transmissionMap.png"
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {figpath}")

    plt.show()


def P_prodVSimport_piechart(df, n_hours, n_scen, savefigure=False, figurename=None, results_dir=None):
    # Velg kolonner vi trenger
    cols = ["Node", "Power generation [MWh]", "Power transmission in [MWh]"]

    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    # Summer opp per land over alle perioder, timer, scenarioer og sesonger
    summary = df[cols].groupby("Node").sum() * 5 * seasonScale / n_scen

    # ---------- 2) Land-koordinater (lon, lat) ----------
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
        "Finland": (25.0, 64.0)
    }

    # Behold bare land vi har koordinater for
    summary = summary.loc[summary.index.intersection(node_coords.keys())].copy()
    summary["total"] = summary["Power generation [MWh]"] + summary["Power transmission in [MWh]"]

    # ----- 3) Pie-tegner som funker i Cartopy-akse -----
    def draw_pie(ax, lon, lat, values, radius_deg, colors=("tab:blue", "tab:orange")):
        """
        Tegn et piechart ved (lon, lat) i PlateCarree-koordinater.
        radius_deg: radius i grader (ca.). NB: litt visuelt skjevt pga. projeksjon, men funker bra.
        """
        total = float(np.sum(values))
        if total <= 0:
            return
        fracs = np.array(values) / total
        start = 0.0
        for frac, color in zip(fracs, colors):
            if frac <= 0:
                continue
            theta1, theta2 = 360 * start, 360 * (start + frac)
            wedge = Wedge((lon, lat), radius_deg, theta1, theta2,
                          facecolor=color, edgecolor="black", linewidth=0.3,
                          transform=ccrs.PlateCarree())
            ax.add_patch(wedge)
            start += frac
        # Tynn kant rundt hele kaka
        ring = Circle((lon, lat), radius_deg, facecolor="none",
                      edgecolor="black", linewidth=0.3, transform=ccrs.PlateCarree())
        ax.add_patch(ring)

    # ----- 4) Figur og bakgrunnskart (Cartopy) -----
    fig = plt.figure(figsize=(14, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-11, 35, 35, 71], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7)

    # ----- 5) Skaler piiestørrelse med total energi -----
    # Bruk radius ~ sqrt(total) slik at AREAL ~ total (mer korrekt visuell skalering)
    t = summary["total"].values
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    r_min, r_max = 0.3, 2.5  # radius i grader (juster etter smak)

    def scale_radius(total):
        if tmax <= 0:
            return (r_min + r_max) / 2
        # sqrt- skalering
        s = np.sqrt(total / tmax)
        return r_min + s * (r_max - r_min)

    # ----- 6) Tegn pie for hvert land -----
    for node, row in summary.iterrows():
        lon, lat = node_coords[node]
        r = scale_radius(row["total"])
        draw_pie(ax, lon, lat, [row["Power generation [MWh]"], row["Power transmission in [MWh]"]], radius_deg=r,
                 colors=("#4C78A8", "#F58518"))  # blå=produksjon, oransje=import

    # ----- 7) Legender -----
    # Fargeforklaring
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Local power production',
               markerfacecolor="#4C78A8", markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Power import',
               markerfacecolor="#F58518", markersize=12),
    ]

    ax.legend(handles=legend_elems, loc="upper left", title="Distribution [%]", fontsize=19, title_fontsize=22)
    plt.tight_layout()

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_prodVSimp_el.png"
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {figpath}")

    plt.show()


def plot_top_map(
        df,
        top_n,
        savefigure=False,
        figurename=None,
        results_dir=None,
):
    """
    Plotter kakediagram-kart med topp-N teknologier per land + 'Other' (total - toppN).
    Alt beregnes internt. Krever cartopy og matplotlib.

    """

    from matplotlib.lines import Line2D

    # --- Koordinater (default) ---

    node_coords = {
        "Austria": (14.55, 47.59), "Belgium": (4.47, 50.85), "BosniaH": (17.67, 43.92),
        "Bulgaria": (25.48, 42.73), "Croatia": (15.98, 45.10), "CzechR": (15.47, 49.74),
        "Denmark": (10.0, 56.0), "France": (2.21, 46.22), "Germany": (10.45, 51.16),
        "GreatBrit.": (-2, 53), "Greece": (21.82, 39.07), "Hungary": (19.40, 47.16),
        "Italy": (12.57, 42.83), "Luxemb.": (6.13, 49.61), "Macedonia": (21.75, 41.61),
        "Netherlands": (5.29, 52.13), "NO1": (10.98, 60.62), "NO2": (7.38, 59.15),
        "NO3": (8.0, 62.47), "NO4": (19.0, 69.0), "NO5": (6.52, 60.57), "Poland": (19.14, 52.13),
        "Portugal": (-8.0, 39.5), "Romania": (24.96, 45.94), "Serbia": (20.45, 44.82),
        "Slovakia": (19.70, 48.66), "Slovenia": (14.51, 46.15), "Spain": (-3.7, 40.4),
        "Sweden": (15.00, 60.12), "Switzerland": (8.23, 46.80), "Ireland": (-8, 53.35),
        "Estonia": (25.0, 58.6), "Latvia": (24.1, 56.9), "Lithuania": (24.0, 55.3),
        "Finland": (25.0, 64.0)
    }

    node_col = "Node"
    tech_col = "GeneratorType"
    value_col = "genExpectedAnnualProduction_GWh"

    tech_colors = {
        'Bio': 'darkslategrey',
        'Bioexisting': 'mediumaquamarine',
        'Coal': 'teal',
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

    # Sett farge for "Other"
    tech_colors["Other"] = "lightgrey"

    # --- Aggreger: total per land og per teknologi ---
    summary = (
        df.groupby([node_col, tech_col], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "GWh"})
    )
    summary = summary[summary[node_col].isin(node_coords.keys())].copy()

    total_per_node = (
        summary.groupby(node_col, as_index=False)["GWh"]
        .sum()
        .rename(columns={"GWh": "GWh_total"})
    )

    # --- Topp-N per land ---
    topN = (
        summary.sort_values([node_col, "GWh"], ascending=[True, False])
        .groupby(node_col, group_keys=False)
        .head(top_n)
    )
    topN_sum = (
        topN.groupby(node_col, as_index=False)["GWh"]
        .sum()
        .rename(columns={"GWh": "GWh_topN"})
    )

    mix = total_per_node.merge(topN_sum, on=node_col, how="left").fillna({"GWh_topN": 0.0})
    mix["GWh_other"] = (mix["GWh_total"] - mix["GWh_topN"]).clip(lower=0.0)

    other_rows = (
        mix[[node_col, "GWh_other"]]
        .rename(columns={"GWh_other": "GWh"})
        .assign(**{tech_col: "Other"})
    )

    topN_plus_other = pd.concat([
        topN[[node_col, tech_col, "GWh"]],
        other_rows
    ], ignore_index=True)

    topN_plus_other = (
        topN_plus_other.sort_values([node_col, "GWh"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # --- Tegnefunksjon ---
    def draw_pie(ax, lon, lat, labels, values, radius_deg):
        tot = float(np.sum(values))
        if tot <= 0:
            return
        fracs = np.array(values, dtype=float) / tot
        start = 0.0
        for lab, frac in zip(labels, fracs):
            if frac <= 0:
                continue
            theta1, theta2 = 360 * start, 360 * (start + frac)
            color = tech_colors.get(lab, "lightgrey")  # fallback
            wedge = Wedge((lon, lat), radius_deg, theta1, theta2,
                          facecolor=color,
                          edgecolor="black", linewidth=0.3,
                          transform=ccrs.PlateCarree())
            ax.add_patch(wedge)
            start += frac
        ring = Circle((lon, lat), radius_deg, facecolor="none",
                      edgecolor="black", linewidth=0.3,
                      transform=ccrs.PlateCarree())
        ax.add_patch(ring)

    extent = (-11, 35, 35, 71)
    r_min = 0.5
    r_max = 2.5
    # --- Radius ~ sqrt(total) ---
    tvals = total_per_node["GWh_total"].values
    tmax = np.nanmax(tvals) if len(tvals) else 0.0

    def scale_radius(total):
        if tmax <= 0:
            return (r_min + r_max) / 2
        s = np.sqrt(total / tmax)
        return r_min + s * (r_max - r_min)

    # --- Figur og kart ---
    fig = plt.figure(figsize=(14, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    x0, x1, y0, y1 = extent
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7)

    for node, g in topN_plus_other.groupby(node_col):
        lon, lat = node_coords[node]
        total = float(total_per_node.loc[total_per_node[node_col] == node, "GWh_total"].values[0])
        r = scale_radius(total)
        labels = g[tech_col].tolist()
        vals = g["GWh"].tolist()
        draw_pie(ax, lon, lat, labels, vals, radius_deg=r)

    # --- Legende ---
    legend_labels = sorted(topN_plus_other[tech_col].unique())
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=lab,
               markerfacecolor=tech_colors.get(lab, "lightgrey"), markersize=10)
        for lab in legend_labels
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=18)

    plt.tight_layout()

    if savefigure:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / f"{figurename}_topTech.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Figur lagret til {out}")

    plt.show()


def HydrogenProd_piechart(df, n_hours, n_scen, savefigure=False, figurename=None, results_dir=None):
    # ---- 0) Sjekk/konverter kolonner ----
    required_cols = ["Node", "PEM production [ton]", "ALK production [ton]", "SOEC production [ton]",
                     "Reformer production [ton]"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i df: {missing}")

    # Sørg for numeriske verdier
    for c in required_cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ---- 1) Aggreger per land ----
    # faktor: (sesongskalering) * (#perioder=5?) / (#scenarier)
    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)
    factor = 5 * seasonScale / float(n_scen)

    summary = (
        df[required_cols]
        .groupby("Node", as_index=True)
        .sum()
        .mul(factor)
    )

    # ---- 2) Koordinater (lon, lat) ----
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
        "GreatBrit.": (-2.0, 53.0),
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
        "Ireland": (-8.0, 53.35),
        "Estonia": (25.0, 58.6),
        "Latvia": (24.1, 56.9),
        "Lithuania": (24.0, 55.3),
        "Finland": (25.0, 64.0),
    }

    # Behold bare land vi har koordinater for
    keep = summary.index.intersection(node_coords.keys())
    missing_nodes = summary.index.difference(keep)
    if len(missing_nodes) > 0:
        print("Advarsel: mangler koordinater for:", ", ".join(missing_nodes))
    summary = summary.loc[keep].copy()

    # Total
    part_cols = ["PEM production [ton]", "ALK production [ton]", "SOEC production [ton]", "Reformer production [ton]"]
    summary["total"] = summary[part_cols].sum(axis=1)

    if summary.empty or summary["total"].sum() == 0:
        raise ValueError("Ingen data å plotte (summary er tom eller totaler=0).")

    # ---- 3) Pie-tegner i Cartopy-akse ----
    def draw_pie(ax, lon, lat, values, radius_deg, colors):
        total = float(np.sum(values))
        if total <= 0:
            return
        fracs = np.array(values, dtype=float) / total
        start = 0.0
        for frac, color in zip(fracs, colors):
            if frac <= 0:
                continue
            theta1, theta2 = 360 * start, 360 * (start + frac)
            wedge = Wedge((lon, lat), radius_deg, theta1, theta2,
                          facecolor=color, edgecolor="black", linewidth=0.3,
                          transform=ccrs.PlateCarree())
            ax.add_patch(wedge)
            start += frac
        ax.add_patch(Circle((lon, lat), radius_deg, facecolor="none",
                            edgecolor="black", linewidth=0.3, transform=ccrs.PlateCarree()))

    # ---- 4) Kart ----
    fig = plt.figure(figsize=(14, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-11, 35, 35, 71], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7)

    # ---- 5) Skaler piestørrelse med total (areal ~ total) ----
    t = summary["total"].values
    tmax = np.nanmax(t)
    r_min, r_max = 0.6, 2.5  # juster etter smak

    def scale_radius(total):
        if tmax <= 0:
            return (r_min + r_max) / 2
        return r_min + np.sqrt(total / tmax) * (r_max - r_min)

    colors = ("steelblue", "plum", "rebeccapurple", "grey")

    # ---- 6) Tegn pie for hvert land ----
    for node, row in summary.iterrows():
        lon, lat = node_coords[node]
        r = scale_radius(row["total"])
        draw_pie(ax, lon, lat, row[part_cols].tolist(), radius_deg=r, colors=colors)

    # ---- 7) Legende ----
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='PEM', markerfacecolor=colors[0], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Alkaline', markerfacecolor=colors[1], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='SOEC', markerfacecolor=colors[2], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Reformer', markerfacecolor=colors[3], markersize=12),
    ]
    ax.legend(handles=legend_elems, loc="upper left", title="Hydrogen technology", fontsize=19, title_fontsize=22)

    plt.tight_layout()

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_H2prod_piechart.png"
        plt.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {figpath}")

    plt.show()


def H2prod_per_country(df, n_hours, n_scen, savefigure=False, figurename=None, results_dir=None):
    required_cols = ["Node", "PEM production [ton]", "ALK production [ton]", "SOEC production [ton]",
                     "Reformer production [ton]"]

    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)
    factor = 5 * seasonScale / float(n_scen)

    summary = (
        df[required_cols]
        .groupby("Node", as_index=True)
        .sum()
        .mul(factor)
    )

    part_cols = ["PEM production [ton]", "ALK production [ton]", "SOEC production [ton]", "Reformer production [ton]"]
    summary["total"] = summary[part_cols].sum(axis=1)

    # 4) Sorter (størst øverst i plottet)
    summary = summary.sort_values("total", ascending=True)

    # >>> Behold kun de 23 største landene <<<
    summary = summary.tail(23)

    scale, unit = 1e6, "[M ton]"

    plot_vals = summary["total"] / scale

    # 6) Plot

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(summary.index, plot_vals, color="teal")

    ax.set_xlabel(f"Total hydrogen production {unit}", fontsize=24)
    ax.set_ylabel("Country", fontsize=24)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    ax.tick_params(axis="x", labelsize=20)  # x-aksen ticks
    ax.tick_params(axis="y", labelsize=20)

    # Verdilapper
    xpad = plot_vals.max() * 0.01 if plot_vals.max() > 0 else 0.05
    for bar, val in zip(bars, plot_vals):
        ax.text(bar.get_width() + xpad, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=14)

    plt.tight_layout()

    if savefigure and results_dir and figurename:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / f"{figurename}_H2prod_country_bar.eps"
        plt.savefig(out, format='eps', bbox_inches='tight')
        print(f"Figure saved to {out}")

    plt.show()


def weather_data_plot(df, scen, country, tech1, tech2=None, tech3=None, savefigure=False, results_dir=None,
                      figurename=None):
    if tech2 and tech3:
        data1 = df[(df["IntermitentGenerators"] == tech1) & (df['Scenario'] == scen) & (df['Node'] == country)]
        data2 = df[(df["IntermitentGenerators"] == tech2) & (df['Scenario'] == scen) & (df['Node'] == country)]
        data3 = df[(df["IntermitentGenerators"] == tech3) & (df['Scenario'] == scen) & (df['Node'] == country)]
        plt.figure(figsize=(14, 10))
        plt.plot(data1['GeneratorStochasticAvailabilityRaw'].to_numpy(), label=tech1, color='green')
        plt.plot(data2['GeneratorStochasticAvailabilityRaw'].to_numpy(), label=tech2, linewidth=2,
                 color='rebeccapurple')
        plt.plot(data3['GeneratorStochasticAvailabilityRaw'].to_numpy(), label=tech3)
        plt.grid(axis='y', alpha=0.6)
        plt.tick_params(axis="x", labelsize=16)
        plt.tick_params(axis="y", labelsize=16)
        plt.legend(fontsize=16)

        if savefigure and results_dir and figurename:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            out = Path(results_dir) / f"{figurename}_weather_input.eps"
            plt.savefig(out, format='eps', bbox_inches='tight')
            print(f"Figure saved to {out}")

        plt.show()


    elif tech2:
        data1 = df[(df["IntermitentGenerators"] == tech1) & (df['Scenario'] == scen) & (df['Node'] == country)]
        data2 = df[(df["IntermitentGenerators"] == tech2) & (df['Scenario'] == scen) & (df['Node'] == country)]
        plt.figure(figsize=(14, 10))
        plt.plot(data1['GeneratorStochasticAvailabilityRaw'].to_numpy(), label=tech1)
        plt.plot(data2['GeneratorStochasticAvailabilityRaw'].to_numpy(), label=tech2)
        plt.grid(axis='y', alpha=0.6)
        plt.tick_params(axis="x", labelsize=16)
        plt.tick_params(axis="y", labelsize=16)
        plt.legend(fontsize=16)
        if savefigure and results_dir and figurename:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            out = Path(results_dir) / f"{figurename}_weather_input.eps"
            plt.savefig(out, format='eps', bbox_inches='tight')
            print(f"Figure saved to {out}")
        plt.show()

    else:
        data1 = df[(df["IntermitentGenerators"] == tech1) & (df['Scenario'] == scen) & (df['Node'] == country)]
        plt.figure(figsize=(14, 10))
        plt.plot(data1['GeneratorStochasticAvailabilityRaw'].to_numpy())
        plt.grid(axis='y', alpha=0.6)
        plt.tick_params(axis="x", labelsize=16)
        plt.tick_params(axis="y", labelsize=16)
        plt.legend(fontsize=16)
        if savefigure and results_dir and figurename:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            out = Path(results_dir) / f"{figurename}_weather_input.eps"
            plt.savefig(out, format='eps', bbox_inches='tight')
            print(f"Figure saved to {out}")
        plt.show()


def plot_hydrogen_use(hydrogen_use, n_hours, n_scen, savefigure=False, figurename=None, results_dir=None):
    hydrogen_use['Total hydrogen demand [ton]'] = hydrogen_use['Hydrogen used for transport [ton]'] + hydrogen_use[
        'Hydrogen used for oil refining [ton]'] + hydrogen_use['Hydrogen used for ammonia [ton]'] + hydrogen_use[
                                                      'Hydrogen used for cement [ton]'] + hydrogen_use[
                                                      'Hydrogen burned for power and heat [ton]'] + hydrogen_use[
                                                      'Hydrogen used for steel [ton]']
    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_scen)

    # -----------------------PIE HYDROGEN TOP 8 NODES----------------------#
    Annual_hydrogen_demand = (hydrogen_use.groupby(['Node'], as_index=False)['Total hydrogen demand [ton]'].sum())
    Annual_hydrogen_demand['Total hydrogen demand [ton]'] = Annual_hydrogen_demand[
                                                                'Total hydrogen demand [ton]'] * seasonScale * 5 / n_scen
    top_8 = Annual_hydrogen_demand.nlargest(8, 'Total hydrogen demand [ton]').reset_index(drop=True)

    other = Annual_hydrogen_demand['Total hydrogen demand [ton]'].sum() - top_8['Total hydrogen demand [ton]'].sum()

    labels = top_8['Node']
    values = top_8['Total hydrogen demand [ton]']

    values_all = list(values) + [other]
    labels_all = list(labels.astype(str)) + ['Other']
    colors = ["teal", "lightskyblue", "darkseagreen", "khaki", "plum", "darkslateblue", "lightsteelblue", "orange",
              "lemonchiffon"]
    plt.figure()
    plt.pie(values_all, labels=labels_all, autopct='%1.1f%%', startangle=90, counterclock=False, colors=colors[:9])
    plt.axis('equal')
    if savefigure:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / f"{figurename}_pieH2TopNodes.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Figur lagret til {out}")
    plt.show()

    # ---------------TOP 8 HYDROGEN DEMAND NODES OVER PERIODS-----------------------#
    Period_hydrogen_demand = (
        hydrogen_use.groupby(['Node', 'Period'], as_index=False)['Total hydrogen demand [ton]'].sum())
    Period_hydrogen_demand['Total hydrogen demand [ton]'] = Period_hydrogen_demand[
                                                                'Total hydrogen demand [ton]'] * seasonScale * 5 / n_scen
    Period_hydrogen_demand = (
        Period_hydrogen_demand[Period_hydrogen_demand['Node'].isin(top_8['Node'])].reset_index(drop=True))

    p = (Period_hydrogen_demand
         .pivot(index='Period', columns='Node', values='Total hydrogen demand [ton]')
         .sort_index()
         .reindex(columns=top_8['Node']))  # Mt

    (p / 1e6).plot(marker='o', color=colors[:8], figsize=(12, 8))
    plt.ylabel('Hydrogen demand M ton', fontsize=16)
    plt.xlabel('Period', fontsize=16)
    plt.xticks(rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.6, linestyle='--')
    plt.legend(fontsize=14)
    plt.tight_layout()
    if savefigure:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / f"{figurename}_developH2TopNodes.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Figur lagret til {out}")
    plt.show()

    # ---------------------HYDROGEN DEVELOPMENT OVER PERIODS BY SECTOR---------------#

    sectors = ['Hydrogen used for transport [ton]', 'Hydrogen used for oil refining [ton]',
               'Hydrogen used for ammonia [ton]', 'Hydrogen used for cement [ton]',
               'Hydrogen burned for power and heat [ton]', 'Hydrogen used for steel [ton]']

    hydrogen_use_sector = (hydrogen_use.groupby(['Period'], as_index=False)[sectors].sum())
    hydrogen_use_sector[sectors] = hydrogen_use_sector[sectors] * seasonScale * 5 / (n_scen * 1e6)
    hydrogen_use_sector.sort_values('Period')
    colors2 = ['palegreen', 'teal', 'gold', 'skyblue', 'orange', 'royalblue']

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(sectors):
        plt.plot(hydrogen_use_sector['Period'], hydrogen_use_sector[col], label=col, marker='o', color=colors2[i])

    plt.ylabel('Hydrogen demand M ton', fontsize=16)
    plt.xlabel('Period', fontsize=16)
    plt.xticks(rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.6, linestyle='--')
    plt.legend(fontsize=14)
    plt.tight_layout()
    if savefigure:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / f"{figurename}_developH2sectors.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Figur lagret til {out}")
    plt.show()

    # --------------------HYDROGEN SECTOR PIE PLOT----------------------#

    hydrogen_use_sector_total = hydrogen_use_sector[sectors].sum()
    short_labels = [' '.join(i.split()[-2:]) for i in hydrogen_use_sector_total.index]
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        hydrogen_use_sector_total,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=colors2,
        radius=0.9
    )
    for t in autotexts:
        t.set_fontsize(12)
    plt.legend(
        wedges,  # bruk wedge-objektene (så fargene matcher)
        short_labels,
        fontsize=14,
        title_fontsize=14,
        title="Hydrogen used for",
        loc="lower left",
        bbox_to_anchor=(0, 0)  # flytt legend ut til høyre
    )
    plt.tight_layout()
    if savefigure:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / f"{figurename}_pieH2sectors.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Figur lagret til {out}")
    plt.show()
