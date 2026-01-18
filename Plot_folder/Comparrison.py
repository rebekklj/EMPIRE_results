from pathlib import Path
import pandas as pd
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
project_dir = Path(__file__).resolve().parents[1]  # EMPIRE_results_git
data_dir = project_dir / "data"

result_dir1 = data_dir / "Results_woH2_storage" / "full_model_base"
result_dir2 = data_dir / "Results_FINAL_BASE_FLEX_emcap_cyclelim" / "full_model_base"
result_dir3 = data_dir / "Results_FINAL_BASE_NOFLEX_emcap_opt" / "full_model_base"
result_dir4 = data_dir / "Results_FINAL_BASE_FLEX_emcap_opt" / "full_model_base"


# -----------------------------
# Constants / helpers
# -----------------------------
PERIODS_ORDER = ["2020-2025", "2025-2030", "2030-2035", "2035-2040",
                 "2040-2045", "2045-2050", "2050-2055"]

OBJ_COLORS_11 = [
    "teal", "darkturquoise", "orange", "yellowgreen", "seagreen",
    "plum", "moccasin", "hotpink", "darkslategrey", "brown", "dodgerblue"
]
OBJ_COLORS_12 = OBJ_COLORS_11 + ["darkgrey"]


def _bn(x):
    return x / 1e9


def _period_sum(df, col, denom=1.0):
    return df.groupby("Period")[col].sum() / denom


def _cum_to_periodic(cum_series):
    cum_series = cum_series.sort_index()
    return cum_series.diff().fillna(cum_series)


def _print_series(title, s, unit="", decimals=3):
    s = s.copy()
    try:
        s = s.astype(float)
    except Exception:
        pass
    print(f"\n{title}")
    for k, v in s.items():
        if isinstance(v, (float, np.floating, int, np.integer)):
            print(f"  {k}: {v:.{decimals}f}{unit}")
        else:
            print(f"  {k}: {v}{unit}")


def _print_top(title, s, n=10, unit="", decimals=3):
    s = s.astype(float).sort_values(key=lambda x: x.abs(), ascending=False)
    print(f"\n{title} (topp {min(n, len(s))} etter |verdi|)")
    for k, v in s.head(n).items():
        print(f"  {k}: {v:.{decimals}f}{unit}")


# -----------------------------
# Objective cost decomposition
# -----------------------------
def compute_period_costs(Gen_op, Gen_inv, H2_inv, Stor_el, Trans_inv, OffConv_inv, OBJ_val, H2_op, n_scen):
    # Generator drift (diskontert i input, skaleres med n_scen)
    gen_op_period = _bn(_period_sum(Gen_op, "OperationalCost_Euro", denom=n_scen))

    # Generator investering
    gen_inv_period = _bn(_period_sum(Gen_inv, "genInvestedCost_Euro"))

    # Offshore converter inv
    offConv_period = _bn(_period_sum(OffConv_inv, "offshoreConversionInvestedCost_Euro"))

    # Industri operational (hydrogen related op cost)
    H2_op_period = _bn(_period_sum(H2_op, "HydrogenRelatedOperationalCost_Euro", denom=n_scen))

    # Hydrogen: kumulative -> periodiske (produksjonsinvest)
    h2_prod_cols = [
        "Discounted PEM_yellow cost [EUR]",
        "Discounted PEM_green cost [EUR]",
        "Discounted PEM_import cost [EUR]",
        "Discounted ALK cost [EUR]",
        "Discounted SOEC cost [EUR]",
        "Discounted Reformer cost [EUR]",
    ]
    h2_prod_cum = H2_inv.groupby("Period")[h2_prod_cols].sum().sum(axis=1)
    h2_prod_period = _bn(_cum_to_periodic(h2_prod_cum))

    # Hydrogen pipeline (kumulativ -> periodisk)
    h2_pipe_cum = H2_inv.groupby("Period")["Discounted pipeline cost [EUR]"].sum()
    h2_pipe_period = _bn(_cum_to_periodic(h2_pipe_cum))

    # Repurposed H2 pipe + CO2 pipe
    re_h2_pipe_period = _bn(_period_sum(Trans_inv, "RepurposedPipeilineInvCost"))
    CO2_pipe_period = _bn(_period_sum(Trans_inv, "CO2PipelineInvCost"))

    # Hydrogen storage (kumulativ -> periodisk)
    h2_stor_cum = H2_inv.groupby("Period")["Discounted storage cost [EUR]"].sum()
    h2_stor_period = _bn(_cum_to_periodic(h2_stor_cum))

    # Power storage & transmission inv
    pstor_period = _bn(_period_sum(Stor_el, "storInvestedCost_Euro"))
    ptrans_period = _bn(_period_sum(Trans_inv, "TransmissionInvCost"))

    # OBJ (bn EUR)
    OBJ_value = (
        OBJ_val.loc[OBJ_val["key"].str.contains("Scientific notation", na=False), "value"].iloc[0]
    ) / 1e9

    df = pd.DataFrame({
        "Generator operational": gen_op_period,
        "Generator investment": gen_inv_period,
        "Hydrogen production inv.": h2_prod_period,
        "Hydrogen pipeline inv.": h2_pipe_period,
        "Hydrogen repurposed pipeline": re_h2_pipe_period,
        "Hydrogen storage inv.": h2_stor_period,
        "Industry operational": H2_op_period,
        "Power storage inv.": pstor_period,
        "Power transmission inv.": ptrans_period,
        "CO2 pipeline inv.": CO2_pipe_period,
        "Offshore converter inv.": offConv_period,
    })

    # ---- Print tall per scenario (nyttig for rapport) ----
    comp = df.sum(axis=0)
    other = OBJ_value - comp.sum()
    print("\n==============================")
    print("compute_period_costs: nivå (bn EUR)")
    print(f"  OBJ total: {OBJ_value:.3f} bn EUR")
    print(f"  Sum komponenter: {comp.sum():.3f} bn EUR")
    print(f"  Other (=OBJ - sum): {other:.3f} bn EUR")
    _print_top("  Komponent-summer", comp, n=20, unit=" bn EUR", decimals=3)

    # ---- Plot: én stacked stolpe for total (komponent-summer + Other) ----
    comp["Other"] = other
    components = comp.index.tolist()
    values = comp.values

    fig, ax = plt.subplots(figsize=(8, 6))
    x = 0.0
    bar_width = 0.5
    bottom_pos, bottom_neg = 0.0, 0.0

    for name, val, col in zip(components, values, OBJ_COLORS_12):
        if np.isclose(val, 0.0):
            continue
        bottom = bottom_pos if val > 0 else bottom_neg
        ax.bar(x, val, width=bar_width, bottom=bottom, color=col,
               edgecolor="black", linewidth=0.5, label=name)
        if val > 0:
            bottom_pos += val
        else:
            bottom_neg += val

    ax.set_xlim(-1, 1)
    ax.set_xticks([x])
    plt.yticks(fontsize=16)
    ax.axhline(0, color="black", linewidth=1)
    plt.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylabel("Objective value [bn EUR]", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title="Components", fontsize=14, title_fontsize=16,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    return df, OBJ_value


def plot_OBJ_generator_inv_diff(df_base, df_scen, name_base="BASE_MOD_NOFLEX", name_scen="BASE_MOD_FLEX"):
    # Align periods
    df_base = df_base.reindex(PERIODS_ORDER).fillna(0)
    df_scen = df_scen.reindex(PERIODS_ORDER).fillna(0)

    df_diff = df_scen - df_base  # scenario - base

    # ---- Print tall for rapport ----
    print("\n==============================")
    print(f"OBJ-komponenter per periode: {name_scen} – {name_base} (bn EUR)")
    _print_series("Total diff per periode (sum over komponenter)",
                  df_diff.sum(axis=1), unit=" bn EUR", decimals=3)
    _print_top("Total diff per komponent (sum over perioder)",
               df_diff.sum(axis=0), n=50, unit=" bn EUR", decimals=3)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(PERIODS_ORDER))
    bottom_pos = np.zeros(len(PERIODS_ORDER))
    bottom_neg = np.zeros(len(PERIODS_ORDER))

    for col, color in zip(df_diff.columns, OBJ_COLORS_11):
        vals = df_diff[col].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        ax.bar(x, pos, bottom=bottom_pos, color=color, label=col)
        bottom_pos += pos
        ax.bar(x, neg, bottom=bottom_neg, color=color)
        bottom_neg += neg

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS_ORDER, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_ylim([-130, 130])
    ax.set_xlabel("Period", fontsize=16)
    ax.set_ylabel("Cost difference [bn EUR]", fontsize=16)
    ax.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.02, 1.0))
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_total_objective_difference(df_base, OBJ_base, df_scen, OBJ_scen,
                                    name_base="BASE", name_scen="SCENARIO"):
    df_diff = df_scen - df_base
    comp_diff = df_diff.sum(axis=0)  # bn EUR
    obj_total_diff = OBJ_scen - OBJ_base
    other = obj_total_diff - comp_diff.sum()
    comp_diff = comp_diff.copy()
    comp_diff["Other"] = other

    # ---- Print tall for rapport ----
    print("\n==============================")
    print(f"Total objective diff: {name_scen} – {name_base} (bn EUR)")
    print(f"  OBJ_scen: {OBJ_scen:.3f}")
    print(f"  OBJ_base: {OBJ_base:.3f}")
    print(f"  ΔOBJ:     {obj_total_diff:.3f}")
    print(f"  Sum(Δkomponenter uten Other): {comp_diff.drop('Other').sum():.3f}")
    print(f"  Other: {other:.3f}")
    _print_top("Δ per komponent (inkl Other)", comp_diff, n=50, unit=" bn EUR", decimals=3)

    # ---- Plot ----
    components = comp_diff.index.tolist()
    values = comp_diff.values

    fig, ax = plt.subplots(figsize=(8, 6))
    x = 0.0
    bar_width = 0.5
    bottom_pos, bottom_neg = 0.0, 0.0

    for name, val, col in zip(components, values, OBJ_COLORS_12):
        if np.isclose(val, 0.0):
            continue
        bottom = bottom_pos if val > 0 else bottom_neg
        ax.bar(x, val, width=bar_width, bottom=bottom, color=col,
               edgecolor="black", linewidth=0.5, label=name)
        if val > 0:
            bottom_pos += val
        else:
            bottom_neg += val

    ax.set_xlim(-1, 1)
    ax.set_xticks([x])
    plt.yticks(fontsize=16)
    ax.axhline(0, color="black", linewidth=1)
    plt.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylabel("Objective value difference [bn EUR]", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title="Components", fontsize=14, title_fontsize=16,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    return comp_diff


# -----------------------------
# Loading results
# -----------------------------
def load_results(result_dir: Path):
    Gen_op = pd.read_csv(
        result_dir / "results_objective_components_operational_costs.csv",
        skiprows=list(range(65353, 65369)),
    )
    H2_op = pd.read_csv(
        result_dir / "results_objective_components_operational_costs.csv",
        skiprows=65354,
    )
    Gen_inv = pd.read_csv(
        result_dir / "results_objective_components_generation_inv_costs.csv",
        skiprows=list(range(5447, 5456)),
    )
    OffConv_inv = pd.read_csv(
        result_dir / "results_objective_components_generation_inv_costs.csv",
        skiprows=5448,
    )
    H2_inv = pd.read_csv(result_dir / "results_hydrogen_costs.csv")
    Stor_el = pd.read_csv(result_dir / "results_objective_components_storage_inv_costs.csv")
    Trans_inv = pd.read_csv(result_dir / "results_transmission_inv_costs.csv")
    OBJ_val = pd.read_csv(
        result_dir / "results_objective.csv",
        sep=",",
        header=None,
        names=["key", "value"],
        skipinitialspace=True,
    )
    OBJ_val["value"] = pd.to_numeric(OBJ_val["value"], errors="coerce")
    return Gen_op, Gen_inv, H2_inv, Stor_el, Trans_inv, OffConv_inv, OBJ_val, H2_op


# -----------------------------
# Production / capacity tech colors
# -----------------------------
tech_colors = {
    "Bio": "darkslategrey",
    "Bioexisting": "mediumaquamarine",
    "Coal": "teal",
    "Coalexisting": "black",
    "GasCCGT": "coral",
    "GasOCGT": "skyblue",
    "Gasexisting": "royalblue",
    "Geo": "steelblue",
    "HydrogenCCGT": "moccasin",
    "HydrogenOCGT": "orange",
    "Hydroregulated": "khaki",
    "Hydrorun-of-the-river": "yellowgreen",
    "Liginiteexisting": "maroon",
    "Lignite": "sienna",
    "LigniteCCSadv": "chocolate",
    "Nuclear": "pink",
    "Oilexisting": "purple",
    "Solar": "violet",
    "Waste": "navy",
    "Wave": "darkslateblue",
    "Windoffshorefloating": "slateblue",
    "Windoffshoregrounded": "lightsteelblue",
    "Windonshore": "seagreen",
}


def compute_production(df):
    work = df.copy()
    work["genExpectedAnnualProduction_GWh"] = pd.to_numeric(
        work["genExpectedAnnualProduction_GWh"], errors="coerce"
    )

    prod = work.pivot_table(
        index="GeneratorType",
        columns="Period",
        values="genExpectedAnnualProduction_GWh",
        aggfunc="sum",
        fill_value=0,
    )

    def _get_year(p):
        m = re.match(r"(\d{4})", str(p))
        return int(m.group(1)) if m else 9999

    periods = sorted(prod.columns, key=_get_year)
    return prod[periods], periods


def plot_production_diff(df_base, df_scen, name_base="BASE", name_scen="SCEN"):
    prod_base, periods = compute_production(df_base)
    prod_scen, _ = compute_production(df_scen)

    all_techs = sorted(set(prod_base.index) | set(prod_scen.index))
    prod_base = prod_base.reindex(all_techs).fillna(0)
    prod_scen = prod_scen.reindex(all_techs).fillna(0)

    prod_diff_TWh = (prod_scen - prod_base) / 1e3  # TWh (annual)

    # ---- Print tall for rapport ----
    print("\n==============================")
    print(f"Årlig produksjonsdiff per periode: {name_scen} – {name_base} (TWh/år)")
    _print_series("Total diff per periode (sum over tech)", prod_diff_TWh.sum(axis=0), unit=" TWh/år", decimals=2)
    _print_top("Total diff per tech (sum over perioder)", prod_diff_TWh.sum(axis=1), n=20, unit=" TWh/år", decimals=2)

    # ---- Plot ----
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(16, 10))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for tech in all_techs:
        vals = prod_diff_TWh.loc[tech].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        color = tech_colors.get(tech, "lightgray")

        ax.bar(x, pos, bottom=bottom_pos, color=color, edgecolor="black", linewidth=0.3)
        bottom_pos += pos
        ax.bar(x, neg, bottom=bottom_neg, color=color, edgecolor="black", linewidth=0.3)
        bottom_neg += neg

    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_ylim([-900, 900])
    plt.yticks(fontsize=18)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=18)
    ax.set_ylabel("Annual production [TWh]", fontsize=20)

    handles = [plt.Rectangle((0, 0), 1, 1, color=tech_colors.get(t, "lightgray")) for t in all_techs]
    ax.legend(handles, all_techs, title="Technology", fontsize=15, title_fontsize=15, ncol=2,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

    return prod_diff_TWh


def compute_production_by_country(df, country_col="Node"):
    work = df.copy()
    work["genExpectedAnnualProduction_GWh"] = pd.to_numeric(
        work["genExpectedAnnualProduction_GWh"], errors="coerce"
    ).fillna(0)

    # tech x land (GWh)
    return (
        work.groupby([country_col, "GeneratorType"])["genExpectedAnnualProduction_GWh"]
        .sum()
        .unstack("GeneratorType", fill_value=0)
        .T
    )


def plot_production_diff_total_by_country(
    df_base, df_scen,
    country_col="Node",
    name_base="BASE", name_scen="SCEN",
    top_n=None,
    sort_by_abs_total=True,
    figsize=(16, 8),
    ylim=None
):
    prod_base = compute_production_by_country(df_base, country_col=country_col)
    prod_scen = compute_production_by_country(df_scen, country_col=country_col)

    all_techs = sorted(set(prod_base.index) | set(prod_scen.index))
    all_ctry = sorted(set(prod_base.columns) | set(prod_scen.columns))

    prod_base = prod_base.reindex(index=all_techs, columns=all_ctry).fillna(0)
    prod_scen = prod_scen.reindex(index=all_techs, columns=all_ctry).fillna(0)

    diff_TWh = (prod_scen - prod_base) / 1e3  # tech x land (TWh)

    total_by_country = diff_TWh.sum(axis=0)
    order = (total_by_country.abs() if sort_by_abs_total else total_by_country).sort_values(ascending=False).index
    if top_n is not None:
        order = order[:top_n]

    diff_TWh = diff_TWh[order]
    countries = list(diff_TWh.columns)

    # ---- Print tall for rapport ----
    print("\n==============================")
    print(f"Total produksjonsdiff per land: {name_scen} – {name_base} (TWh, sum over tech og perioder)")
    _print_series("Total diff per land (TWh)", diff_TWh.sum(axis=0), unit=" TWh", decimals=2)
    _print_top("Tech med størst total påvirkning (sum over land i utvalget)", diff_TWh.sum(axis=1), n=20, unit=" TWh", decimals=2)

    # ---- Plot ----
    x = np.arange(len(countries))
    fig, ax = plt.subplots(figsize=figsize)

    bottom_pos = np.zeros(len(countries))
    bottom_neg = np.zeros(len(countries))
    seen_labels = set()

    for tech in all_techs:
        vals = diff_TWh.loc[tech].values
        if np.all(np.abs(vals) < 1e-12):
            continue

        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        color = tech_colors.get(tech, "lightgray")
        lbl = tech if tech not in seen_labels else None

        ax.bar(x, pos, bottom=bottom_pos, color=color, edgecolor="black", linewidth=0.3, label=lbl)
        ax.bar(x, neg, bottom=bottom_neg, color=color, edgecolor="black", linewidth=0.3)

        bottom_pos += pos
        bottom_neg += neg

        if lbl is not None:
            seen_labels.add(tech)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=16)
    ax.set_ylabel("Total production difference [TWh]", fontsize=16)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        y_min = min(0, bottom_neg.min()) * 1.1
        y_max = max(0, bottom_pos.max()) * 1.1
        ax.set_ylim([y_min if y_min != 0 else -1, y_max if y_max != 0 else 1])

    ax.grid(axis="y")
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left",
              title_fontsize=16, fontsize=14)
    plt.tight_layout()
    plt.show()

    return diff_TWh


def plot_production_total_diff(
    df_base, df_scen, df_base3, df_scen3,
    name_base="BASE", name_scen="SCEN",
    name_base3="BASE3", name_scen3="SCEN3",
    label1=None, label2=None,
    width=0.35
):
    prod_base, _ = compute_production(df_base)
    prod_scen, _ = compute_production(df_scen)
    prod_base3, _ = compute_production(df_base3)
    prod_scen3, _ = compute_production(df_scen3)

    all_techs = sorted(set(prod_base.index) | set(prod_scen.index) | set(prod_base3.index) | set(prod_scen3.index))

    prod_base = prod_base.reindex(all_techs).fillna(0)
    prod_scen = prod_scen.reindex(all_techs).fillna(0)
    prod_base3 = prod_base3.reindex(all_techs).fillna(0)
    prod_scen3 = prod_scen3.reindex(all_techs).fillna(0)

    diff1_TWh = (prod_scen - prod_base) * 5 / 1e3
    diff2_TWh = (prod_scen3 - prod_base3) * 5 / 1e3

    total_diff1 = diff1_TWh.sum(axis=1)
    total_diff2 = diff2_TWh.sum(axis=1)

    eps = 1e-8
    techs_with_diff = [t for t in all_techs if abs(float(total_diff1.loc[t])) > eps or abs(float(total_diff2.loc[t])) > eps]

    label1 = label1 or "Moderate nuclear capex"
    label2 = label2 or "Optimistic nuclear capex"

    # ---- Print tall for rapport ----
    print("\n==============================")
    print("Total produksjonsdiff per teknologi (TWh, sum over alle perioder)")
    _print_top(f"{label1}: tech total", total_diff1, n=25, unit=" TWh", decimals=2)
    _print_top(f"{label2}: tech total", total_diff2, n=25, unit=" TWh", decimals=2)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.array([0, 1])
    bottom_pos = np.zeros(2)
    bottom_neg = np.zeros(2)
    seen_labels = set()

    for tech in techs_with_diff:
        vals = np.array([float(total_diff1.loc[tech]), float(total_diff2.loc[tech])])
        color = tech_colors.get(tech, "lightgray")

        for i, v in enumerate(vals):
            if abs(v) <= eps:
                continue
            lbl = tech if tech not in seen_labels else None
            bottom = bottom_pos[i] if v > 0 else bottom_neg[i]
            ax.bar(x[i], v, bottom=bottom, width=width, color=color, edgecolor="black", linewidth=0.3, label=lbl)
            if v > 0:
                bottom_pos[i] += v
            else:
                bottom_neg[i] += v
            if lbl is not None:
                seen_labels.add(tech)

    ax.axhline(0, color="black")
    ax.set_xticks(x)
    plt.yticks(fontsize=16)
    ax.set_xticklabels([label1, label2], fontsize=14)
    ax.set_ylabel("Total production difference [TWh]", fontsize=20)

    y_min = min(0, bottom_neg.min()) * 1.1
    y_max = max(0, bottom_pos.max()) * 1.1
    ax.set_ylim([y_min if y_min != 0 else -1, y_max if y_max != 0 else 1])
    ax.set_xlim(-0.6, 1.6)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Technology", fontsize=13, title_fontsize=14,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"diff_1_TWh": total_diff1, "diff_2_TWh": total_diff2}, index=all_techs)


# -----------------------------
# Installed capacity
# -----------------------------
def compute_installedCap(df):
    work = df.copy()
    work["genInstalledCap_MW"] = pd.to_numeric(work["genInstalledCap_MW"], errors="coerce")

    cap = work.pivot_table(
        index="GeneratorType",
        columns="Period",
        values="genInstalledCap_MW",
        aggfunc="sum",
        fill_value=0,
    )

    def _get_year(p):
        m = re.match(r"(\d{4})", str(p))
        return int(m.group(1)) if m else 9999

    periods = sorted(cap.columns, key=_get_year)
    return cap[periods], periods


def plot_installedcap_diff(df_base, df_scen, name_base="BASE", name_scen="SCEN"):
    cap_base, periods = compute_installedCap(df_base)
    cap_scen, _ = compute_installedCap(df_scen)

    all_techs = sorted(set(cap_base.index) | set(cap_scen.index))
    cap_base = cap_base.reindex(all_techs).fillna(0)
    cap_scen = cap_scen.reindex(all_techs).fillna(0)

    cap_diff_GW = (cap_scen - cap_base) / 1e3  # MW -> GW

    # ---- Print tall for rapport ----
    print("\n==============================")
    print(f"Installert kapasitet diff: {name_scen} – {name_base} (GW)")
    _print_series("Total diff per periode (sum over tech)", cap_diff_GW.sum(axis=0), unit=" GW", decimals=2)
    _print_top("Total diff per tech (sum over perioder)", cap_diff_GW.sum(axis=1), n=25, unit=" GW", decimals=2)

    # ---- Plot (som før) ----
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(16, 12))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    tol = 1e-12
    techs_with_diff = [t for t in all_techs if np.any(np.abs(cap_diff_GW.loc[t].values) > tol)]

    for tech in techs_with_diff:
        vals = cap_diff_GW.loc[tech].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        color = tech_colors.get(tech, "lightgray")

        ax.bar(x, pos, bottom=bottom_pos, color=color, edgecolor="black", linewidth=0.3, label=tech)
        bottom_pos += pos
        ax.bar(x, neg, bottom=bottom_neg, color=color, edgecolor="black", linewidth=0.3)
        bottom_neg += neg

    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_ylim([-600, 600])
    plt.yticks(fontsize=24)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=26)
    ax.set_ylabel("Installed capacity [GW]", fontsize=26)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)

    ax.legend(uniq_h, uniq_l, title="Technology",
              fontsize=18, title_fontsize=20,
              ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    return cap_diff_GW


# -----------------------------
# Hydrogen production (per tech) diff
# -----------------------------
def Yearly_hydrogenProd_perTech_diff(
    df_base, df_scen,
    x1, x2, x3, x4, x5, x6,
    n_scen, n_hours,
    name_base="FLEX", name_scen="NOFLEX",
    y_lim=None, savefigure=False, results_dir=None, figurename=None
):
    seasonScale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    def _prep(df):
        agg = (
            df.groupby("Period")
            .agg({x1: "sum", x2: "sum", x3: "sum", x4: "sum", x5: "sum", x6: "sum"})
            .reset_index()
        )
        agg["Period"] = pd.Categorical(agg["Period"], categories=PERIODS_ORDER, ordered=True)
        agg = agg.sort_values("Period").set_index("Period")

        agg[[x1, x2, x3, x4, x5, x6]] = agg[[x1, x2, x3, x4, x5, x6]] * seasonScale / n_scen
        return agg.reindex(PERIODS_ORDER).fillna(0)

    agg_base = _prep(df_base)
    agg_scen = _prep(df_scen)
    diff = agg_scen - agg_base

    # ---- Print tall for rapport (Mton/år per periode + totaler) ----
    diff_Mton = diff / 1e6
    print("\n==============================")
    print(f"Årlig H2-produksjonsdiff per periode: {name_scen} – {name_base} (Mton/år)")
    _print_series("Total diff per periode (sum over tech)", diff_Mton.sum(axis=1), unit=" Mton/år", decimals=3)
    _print_top("Total diff per tech (sum over perioder)", diff_Mton.sum(axis=0), n=20, unit=" Mton/år", decimals=3)

    # ---- Plot (som før) ----
    fig, ax = plt.subplots(figsize=(12, 8))
    pos_x = np.arange(len(PERIODS_ORDER)) * 0.5
    width = 0.3

    tech_cols = [x1, x2, x3, x4, x5, x6]
    labels = ["PEM green", "PEM yellow", "PEM import", "Alkaline", "SOEC", "Reformer"]
    colors = ["seagreen", "gold", "orange", "plum", "rebeccapurple", "grey"]

    bottom_pos = np.zeros(len(PERIODS_ORDER))
    bottom_neg = np.zeros(len(PERIODS_ORDER))

    for col, lab, color in zip(tech_cols, labels, colors):
        vals = diff[col].values / 1e6  # Mton
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        ax.bar(pos_x, pos, width, bottom=bottom_pos, color=color, alpha=0.8,
               edgecolor="black", linewidth=0.4, label=lab)
        bottom_pos += pos

        ax.bar(pos_x, neg, width, bottom=bottom_neg, color=color, alpha=0.8,
               edgecolor="black", linewidth=0.4)
        bottom_neg += neg

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(pos_x)
    ax.set_ylim([-20, 20])
    ax.set_xticklabels(PERIODS_ORDER, rotation=0)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    ax.set_xlabel("Investment period", fontsize=18)
    ax.set_ylabel(f"Δ annual hydrogen production [M ton]", fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=16, title="Technology", title_fontsize=18)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if savefigure and figurename and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        figpath = Path(results_dir) / f"{figurename}_AnnH2prod_DIFF.png"
        plt.savefig(figpath, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {figpath}")

    plt.tight_layout()
    plt.show()

    return diff


# -----------------------------
# Hydrogen storage capacity diff
# -----------------------------
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
    if periods is None:
        periods = PERIODS_ORDER

    cols = [
        "Total SaltCavern storage capacity [ton]",
        "Total DGF storage capacity [ton]",
        "Total Aquifer storage capacity [ton]",
    ]
    labels = ["Salt cavern", "DGF", "Aquifer"]
    colors = ["lightskyblue", "orange", "yellowgreen"]

    def _prep(df):
        work = df.copy()
        for c in cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        return work.groupby("Period")[cols].sum().reindex(periods).fillna(0)

    cap_base = _prep(df_base)
    cap_scen = _prep(df_scen)

    diff_Mton = (cap_scen - cap_base) / 1e6

    # ---- Print tall for rapport ----
    print("\n==============================")
    print(f"H2-lagringskapasitet diff: {name_scen} – {name_base} (Mton)")
    _print_series("Total diff per periode (sum over lagringstyper)", diff_Mton.sum(axis=1), unit=" Mton", decimals=3)
    _print_top("Total diff per lagringstype (sum over perioder)", diff_Mton.sum(axis=0), n=10, unit=" Mton", decimals=3)

    # ---- Plot ----
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(12, 7))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for c, lab, col in zip(cols, labels, colors):
        vals = diff_Mton[c].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        ax.bar(x, pos, bottom=bottom_pos, color=col, edgecolor="black", linewidth=0.4, label=lab)
        bottom_pos += pos

        ax.bar(x, neg, bottom=bottom_neg, color=col, edgecolor="black", linewidth=0.4)
        bottom_neg += neg

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel(f"Hydrogen storage capacity difference  [Mton]", fontsize=14)
    ax.set_xlabel("Period", fontsize=14)

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


# ============================================================
# Kjøring (samme som hos deg)
# ============================================================
n_scen = 2

Gen_op1, Gen_inv1, H2_inv1, Stor_el1, Trans_inv1, OffConv_inv1, OBJ_val1, H2_op1 = load_results(result_dir1)
Gen_op2, Gen_inv2, H2_inv2, Stor_el2, Trans_inv2, OffConv_inv2, OBJ_val2, H2_op2 = load_results(result_dir2)

df_base_obj, OBJ_base = compute_period_costs(Gen_op2, Gen_inv2, H2_inv2, Stor_el2, Trans_inv2, OffConv_inv2, OBJ_val2, H2_op2, n_scen)
df_scen_obj, OBJ_scen = compute_period_costs(Gen_op1, Gen_inv1, H2_inv1, Stor_el1, Trans_inv1, OffConv_inv1, OBJ_val1, H2_op1, n_scen)

print("\nOBJ_scen:", OBJ_scen)
print("OBJ_base:", OBJ_base)

plot_OBJ_generator_inv_diff(df_base_obj, df_scen_obj, name_base="BASE_MOD_FLEX", name_scen="BASE_MOD_NOFLEX")

comp_diff = plot_total_objective_difference(
    df_base_obj, OBJ_base,
    df_scen_obj, OBJ_scen,
    name_base="BASE_MOD_FLEX",
    name_scen="BASE_MOD_NOFLEX",
)

# ---- Production/investment csvs ----
Elec_genInv_FLEX = pd.read_csv(result_dir2 / "results_elec_generation_inv.csv")
Elec_genInv_NOFLEX = pd.read_csv(result_dir1 / "results_elec_generation_inv.csv")
Elec_genInv_FLEX3 = pd.read_csv(result_dir4 / "results_elec_generation_inv.csv")
Elec_genInv_NOFLEX3 = pd.read_csv(result_dir3 / "results_elec_generation_inv.csv")

df_base = Elec_genInv_FLEX
df_scen = Elec_genInv_NOFLEX
df_base3 = Elec_genInv_FLEX3
df_scen3 = Elec_genInv_NOFLEX3

prod_diff_TWh = plot_production_diff(df_base=df_base, df_scen=df_scen, name_base="FLEX", name_scen="NOFLEX")

plot_production_diff_total_by_country(
    df_base, df_scen,
    country_col="Node",
    name_base="BASE", name_scen="SCEN",
    top_n=23,
    sort_by_abs_total=True,
    figsize=(16, 8),
    ylim=None,
)

res = plot_production_total_diff(df_base, df_scen, df_base3, df_scen3)

cap_diff_GW = plot_installedcap_diff(df_base=df_base, df_scen=df_scen, name_base="FLEX", name_scen="NOFLEX")

hydrogen_production_FLEX = pd.read_csv(result_dir2 / "results_hydrogen_production.csv")
hydrogen_production_NOFLEX = pd.read_csv(result_dir1 / "results_hydrogen_production.csv")

diff_h2_prod = Yearly_hydrogenProd_perTech_diff(
    df_base=hydrogen_production_FLEX,
    df_scen=hydrogen_production_NOFLEX,
    x1="PEM_green production [ton]",
    x2="PEM_yellow production [ton]",
    x3="PEM_import production [ton]",
    x4="ALK production [ton]",
    x5="SOEC production [ton]",
    x6="Reformer production [ton]",
    n_scen=2,
    n_hours=12,
    name_base="FLEX",
    name_scen="NOFLEX",
)

df_store_flex = pd.read_csv(result_dir2 / "results_hydrogen_storage_inv.csv")
df_store_noflex = pd.read_csv(result_dir1 / "results_hydrogen_storage_inv.csv")

diff_store_Mton = hydrogen_storage_capacity_diff(
    df_base=df_store_flex,
    df_scen=df_store_noflex,
    name_base="FLEX",
    name_scen="NOFLEX",
)






