from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG: scenarios / paths
# =============================================================================
project_dir = Path(__file__).resolve().parents[1]  # EMPIRE_results_git
data_dir = project_dir / "data"

SCENARIOS = {
    # Used in comparisons below
    "NOFLEX": data_dir / "Results_FINAL_BASE_NOFLEX_emcap_cyclelim" / "full_model_base",
    "FLEX":   data_dir / "Results_FINAL_woALKSOEC" / "full_model_base",

    # Optional extra pair in "BSTORAGE vs BFLAT" comparison below
    "NOFLEX3": data_dir / "Results_final_SOECpess" / "full_model_base",
    "FLEX3":   data_dir / "Results_FINAL_woALKSOEC" / "full_model_base",
}

# Primary comparison (used for objective / production / capacity / H2 sections)
BASE_NAME = "FLEX"
SCEN_NAME = "NOFLEX"

# Secondary comparison (used for the two-bar total-production plot)
BASE2_NAME = "FLEX3"
SCEN2_NAME = "NOFLEX3"

# Labels for the two-bar plot (these are display labels, not scenario keys)
LABEL1 = "SOEC moderate"
LABEL2 = "SOEC pessemistic"

# Stochastic scenarios count used to scale operational costs and seasonal scaling
N_SCEN = 2

# Period ordering (kept fixed)
PERIODS_ORDER = [
    "2020-2025", "2025-2030", "2030-2035", "2035-2040",
    "2040-2045", "2045-2050", "2050-2055",
]

# =============================================================================
# PLOT COLORS
# =============================================================================
OBJ_COLORS_11 = [
    "teal", "darkturquoise", "orange", "yellowgreen", "seagreen",
    "plum", "moccasin", "hotpink", "darkslategrey", "brown", "dodgerblue",
]
OBJ_COLORS_12 = OBJ_COLORS_11 + ["darkgrey"]

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


# =============================================================================
# HELPERS: numeric + printing
# =============================================================================
def _bn(x: pd.Series | float) -> pd.Series | float:
    """EUR -> bn EUR"""
    return x / 1e9


def _period_sum(df: pd.DataFrame, col: str, denom: float = 1.0) -> pd.Series:
    return df.groupby("Period")[col].sum() / denom


def _cum_to_periodic(cum_series: pd.Series) -> pd.Series:
    cum_series = cum_series.sort_index()
    return cum_series.diff().fillna(cum_series)


def _align_series(a: pd.Series, b: pd.Series, fill_value: float = 0.0):
    idx = a.index.union(b.index)
    return a.reindex(idx).fillna(fill_value), b.reindex(idx).fillna(fill_value)


def _print_block_header(title: str, *, base_name: str, base_dir: Path, scen_name: str, scen_dir: Path):
    print("\n" + "=" * 80)
    print(title)
    print(f"Comparison: {scen_name} (SCEN)  vs  {base_name} (BASE)")
    print(f"  BASE dir: {base_dir}")
    print(f"  SCEN dir: {scen_dir}")
    print("=" * 80)


def _print_series_expr(
    title: str,
    scen: pd.Series,
    base: pd.Series,
    *,
    unit: str = "",
    decimals: int = 3,
    scen_name: str = "SCEN",
    base_name: str = "BASE",
    fill_value: float = 0.0,
):
    scen = pd.Series(scen).astype(float)
    base = pd.Series(base).astype(float)
    scen, base = _align_series(scen, base, fill_value=fill_value)

    print(f"\n{title}")
    for k in scen.index:
        a = float(scen.loc[k])
        b = float(base.loc[k])
        print(f"  {k}: ({a:.{decimals}f}{unit}) - ({b:.{decimals}f}{unit})   [{scen_name} - {base_name}]")


def _print_top_expr(
    title: str,
    scen: pd.Series,
    base: pd.Series,
    *,
    n: int = 10,
    unit: str = "",
    decimals: int = 3,
    scen_name: str = "SCEN",
    base_name: str = "BASE",
    fill_value: float = 0.0,
):
    scen = pd.Series(scen).astype(float)
    base = pd.Series(base).astype(float)
    scen, base = _align_series(scen, base, fill_value=fill_value)

    diff = scen - base
    order = diff.abs().sort_values(ascending=False).index
    n_show = min(n, len(order))

    print(f"\n{title} (top {n_show} by |diff|)")
    for k in order[:n_show]:
        a = float(scen.loc[k])
        b = float(base.loc[k])
        print(f"  {k}: ({a:.{decimals}f}{unit}) - ({b:.{decimals}f}{unit})   [{scen_name} - {base_name}]")


def _require_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


# =============================================================================
# LOADERS
# =============================================================================
def load_objective_results(result_dir: Path):
    """
    Loads objective decomposition inputs.
    NOTE: Your CSVs appear to contain multiple sections with repeated headers; skiprows are kept as-is.
    """
    _require_exists(result_dir, "result_dir")

    gen_op = pd.read_csv(
        result_dir / "results_objective_components_operational_costs.csv",
        skiprows=list(range(65353, 65369)),
    )
    h2_op = pd.read_csv(
        result_dir / "results_objective_components_operational_costs.csv",
        skiprows=65354,
    )
    gen_inv = pd.read_csv(
        result_dir / "results_objective_components_generation_inv_costs.csv",
        skiprows=list(range(5447, 5456)),
    )
    offconv_inv = pd.read_csv(
        result_dir / "results_objective_components_generation_inv_costs.csv",
        skiprows=5448,
    )
    h2_inv = pd.read_csv(result_dir / "results_hydrogen_costs.csv")
    stor_el = pd.read_csv(result_dir / "results_objective_components_storage_inv_costs.csv")
    trans_inv = pd.read_csv(result_dir / "results_transmission_inv_costs.csv")
    obj_val = pd.read_csv(
        result_dir / "results_objective.csv",
        sep=",",
        header=None,
        names=["key", "value"],
        skipinitialspace=True,
    )
    obj_val["value"] = pd.to_numeric(obj_val["value"], errors="coerce")
    return gen_op, gen_inv, h2_inv, stor_el, trans_inv, offconv_inv, obj_val, h2_op


def load_elec_gen_inv(result_dir: Path) -> pd.DataFrame:
    return pd.read_csv(result_dir / "results_elec_generation_inv.csv")


def load_h2_prod(result_dir: Path) -> pd.DataFrame:
    return pd.read_csv(result_dir / "results_hydrogen_production.csv")


def load_h2_storage_inv(result_dir: Path) -> pd.DataFrame:
    return pd.read_csv(result_dir / "results_hydrogen_storage_inv.csv")


# =============================================================================
# OBJECTIVE: compute + plots
# =============================================================================
def compute_period_costs(
    gen_op: pd.DataFrame,
    gen_inv: pd.DataFrame,
    h2_inv: pd.DataFrame,
    stor_el: pd.DataFrame,
    trans_inv: pd.DataFrame,
    offconv_inv: pd.DataFrame,
    obj_val: pd.DataFrame,
    h2_op: pd.DataFrame,
    n_scen: int,
):
    """
    Returns:
      - df_period: rows=Period, cols=components, values=bn EUR per period
      - OBJ_value: total objective bn EUR (scalar)
    """
    # Generator operational (discounted in file; scale by number of stochastic scenarios)
    gen_op_period = _bn(_period_sum(gen_op, "OperationalCost_Euro", denom=n_scen))

    # Generator investment
    gen_inv_period = _bn(_period_sum(gen_inv, "genInvestedCost_Euro"))

    # Offshore converter investment
    offconv_period = _bn(_period_sum(offconv_inv, "offshoreConversionInvestedCost_Euro"))

    # Industry operational (hydrogen related operational cost)
    h2_op_period = _bn(_period_sum(h2_op, "HydrogenRelatedOperationalCost_Euro", denom=n_scen))

    # H2 production investments: cumulative -> periodic
    h2_prod_cols = [
        "Discounted PEM_yellow cost [EUR]",
        "Discounted PEM_green cost [EUR]",
        "Discounted PEM_import cost [EUR]",
        "Discounted ALK cost [EUR]",
        "Discounted SOEC cost [EUR]",
        "Discounted Reformer cost [EUR]",
    ]
    h2_prod_cum = h2_inv.groupby("Period")[h2_prod_cols].sum().sum(axis=1)
    h2_prod_period = _bn(_cum_to_periodic(h2_prod_cum))

    # H2 pipeline: cumulative -> periodic
    h2_pipe_cum = h2_inv.groupby("Period")["Discounted pipeline cost [EUR]"].sum()
    h2_pipe_period = _bn(_cum_to_periodic(h2_pipe_cum))

    # Repurposed H2 pipeline + CO2 pipeline (already periodic in Trans_inv file)
    re_h2_pipe_period = _bn(_period_sum(trans_inv, "RepurposedPipeilineInvCost"))
    co2_pipe_period = _bn(_period_sum(trans_inv, "CO2PipelineInvCost"))

    # H2 storage: cumulative -> periodic
    h2_stor_cum = h2_inv.groupby("Period")["Discounted storage cost [EUR]"].sum()
    h2_stor_period = _bn(_cum_to_periodic(h2_stor_cum))

    # Power storage & transmission investment
    pstor_period = _bn(_period_sum(stor_el, "storInvestedCost_Euro"))
    ptrans_period = _bn(_period_sum(trans_inv, "TransmissionInvCost"))

    # Total objective (bn EUR)
    obj_value = (
        obj_val.loc[obj_val["key"].str.contains("Scientific notation", na=False), "value"].iloc[0]
    ) / 1e9

    df_period = pd.DataFrame({
        "Generator operational": gen_op_period,
        "Generator investment": gen_inv_period,
        "Hydrogen production inv.": h2_prod_period,
        "Hydrogen pipeline inv.": h2_pipe_period,
        "Hydrogen repurposed pipeline": re_h2_pipe_period,
        "Hydrogen storage inv.": h2_stor_period,
        "Industry operational": h2_op_period,
        "Power storage inv.": pstor_period,
        "Power transmission inv.": ptrans_period,
        "CO2 pipeline inv.": co2_pipe_period,
        "Offshore converter inv.": offconv_period,
    })

    return df_period, float(obj_value)


def plot_objective_single_bar(df_period: pd.DataFrame, obj_value_bn: float, *, title: str):
    """
    One stacked bar showing component sums + 'Other' = OBJ - sum(components).
    """
    comp = df_period.sum(axis=0)  # bn EUR per component (summed over periods)
    other = obj_value_bn - comp.sum()

    # Print (single scenario level)
    print("\n" + "-" * 80)
    print(title)
    print(f"OBJ total: {obj_value_bn:.3f} bn EUR")
    print(f"Sum components: {comp.sum():.3f} bn EUR")
    print(f"Other (=OBJ - sum): {other:.3f} bn EUR")

    comp_plot = comp.copy()
    comp_plot["Other"] = other

    # Plot
    components = comp_plot.index.tolist()
    values = comp_plot.values

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
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Objective value [bn EUR]", fontsize=14)
    ax.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title="Components", fontsize=10, title_fontsize=11,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()


def plot_objective_components_diff(df_base: pd.DataFrame, df_scen: pd.DataFrame, *, base_name: str, scen_name: str):
    """
    Plots diff by period (stacked) but prints levels as "(scen) - (base)".
    """
    df_base = df_base.reindex(PERIODS_ORDER).fillna(0)
    df_scen = df_scen.reindex(PERIODS_ORDER).fillna(0)
    df_diff = df_scen - df_base

    # Print: period totals and component totals as expression
    _print_series_expr(
        "Objective cost per period (sum over components) [bn EUR]",
        df_scen.sum(axis=1), df_base.sum(axis=1),
        unit=" bn EUR", decimals=3,
        scen_name=scen_name, base_name=base_name
    )
    _print_top_expr(
        "Objective component totals (sum over periods) [bn EUR]",
        df_scen.sum(axis=0), df_base.sum(axis=0),
        n=50, unit=" bn EUR", decimals=3,
        scen_name=scen_name, base_name=base_name
    )

    # Plot diff
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
    ax.set_xlabel("Period")
    ax.set_ylabel("Cost difference [bn EUR] (SCEN - BASE)")
    ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_total_objective_difference(
    df_base: pd.DataFrame,
    obj_base_bn: float,
    df_scen: pd.DataFrame,
    obj_scen_bn: float,
    *,
    base_name: str,
    scen_name: str,
):
    """
    Prints component levels as "(scen) - (base)" incl. Other, and plots diff stacked bar.
    """
    # Diff decomposition
    df_diff = df_scen - df_base
    comp_diff = df_diff.sum(axis=0)  # bn EUR
    obj_total_diff = obj_scen_bn - obj_base_bn
    other_diff = obj_total_diff - comp_diff.sum()

    # Levels per scenario including Other
    base_comp = df_base.sum(axis=0)
    scen_comp = df_scen.sum(axis=0)
    base_other = obj_base_bn - base_comp.sum()
    scen_other = obj_scen_bn - scen_comp.sum()

    base_with_other = base_comp.copy()
    scen_with_other = scen_comp.copy()
    base_with_other["Other"] = base_other
    scen_with_other["Other"] = scen_other

    print("\n" + "-" * 80)
    print("Objective totals (bn EUR)")
    print(f"  {scen_name}: {obj_scen_bn:.3f}")
    print(f"  {base_name}: {obj_base_bn:.3f}")
    print(f"  Diff (SCEN - BASE): {obj_total_diff:.3f}")
    print(f"  Implied Other diff: {other_diff:.3f}")

    _print_top_expr(
        "Component levels incl. Other (sum over periods) [bn EUR]",
        scen_with_other, base_with_other,
        n=50, unit=" bn EUR", decimals=3,
        scen_name=scen_name, base_name=base_name
    )

    # Plot diff
    comp_diff_plot = comp_diff.copy()
    comp_diff_plot["Other"] = other_diff

    components = comp_diff_plot.index.tolist()
    values = comp_diff_plot.values

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
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylabel("Objective diff [bn EUR] (SCEN - BASE)", fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title="Components", fontsize=10, title_fontsize=11,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    return comp_diff_plot


# =============================================================================
# ELECTRICITY: production (annual per period) + totals by tech/country
# =============================================================================
def compute_annual_production_gwh(df: pd.DataFrame):
    """
    Returns:
      prod_gwh: index=GeneratorType, columns=Period, values=GWh per year
      periods: sorted periods
    """
    work = df.copy()
    work["genExpectedAnnualProduction_GWh"] = pd.to_numeric(
        work["genExpectedAnnualProduction_GWh"], errors="coerce"
    ).fillna(0)

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


def plot_annual_production_diff(df_base: pd.DataFrame, df_scen: pd.DataFrame, *, base_name: str, scen_name: str):
    """
    Annual production per period (TWh/year):
      - prints totals per period and totals per tech as "(scen) - (base)"
      - plots diff stacked by tech per period
    """
    prod_base_gwh, periods = compute_annual_production_gwh(df_base)
    prod_scen_gwh, _ = compute_annual_production_gwh(df_scen)

    all_techs = sorted(set(prod_base_gwh.index) | set(prod_scen_gwh.index))
    prod_base_gwh = prod_base_gwh.reindex(all_techs).fillna(0)
    prod_scen_gwh = prod_scen_gwh.reindex(all_techs).fillna(0)

    base_twh = prod_base_gwh / 1e3
    scen_twh = prod_scen_gwh / 1e3
    diff_twh = (prod_scen_gwh - prod_base_gwh) / 1e3

    # Prints (levels as expressions)
    _print_series_expr(
        "Electricity: annual production totals per period (sum over tech) [TWh/year]",
        scen_twh.sum(axis=0), base_twh.sum(axis=0),
        unit=" TWh/yr", decimals=2,
        scen_name=scen_name, base_name=base_name
    )
    _print_top_expr(
        "Electricity: annual production totals per technology (sum over periods) [TWh/year]",
        scen_twh.sum(axis=1), base_twh.sum(axis=1),
        n=20, unit=" TWh/yr", decimals=2,
        scen_name=scen_name, base_name=base_name
    )

    # Plot diff
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(16, 10))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for tech in all_techs:
        vals = diff_twh.loc[tech].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        color = tech_colors.get(tech, "lightgray")

        ax.bar(x, pos, bottom=bottom_pos, color=color, edgecolor="black", linewidth=0.3)
        bottom_pos += pos
        ax.bar(x, neg, bottom=bottom_neg, color=color, edgecolor="black", linewidth=0.3)
        bottom_neg += neg

    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("Annual production diff [TWh/year] (SCEN - BASE)", fontsize=14)
    ax.grid(axis="y")

    handles = [plt.Rectangle((0, 0), 1, 1, color=tech_colors.get(t, "lightgray")) for t in all_techs]
    ax.legend(handles, all_techs, title="Technology", fontsize=10, title_fontsize=11, ncol=2,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return diff_twh


def compute_total_production_by_country_gwh(df: pd.DataFrame, *, country_col: str = "Node") -> pd.DataFrame:
    """
    Total (summed across periods) by country & technology:
      returns DataFrame tech x country in GWh (sum over all rows)
    """
    work = df.copy()
    work["genExpectedAnnualProduction_GWh"] = pd.to_numeric(
        work["genExpectedAnnualProduction_GWh"], errors="coerce"
    ).fillna(0)

    return (
        work.groupby([country_col, "GeneratorType"])["genExpectedAnnualProduction_GWh"]
        .sum()
        .unstack("GeneratorType", fill_value=0)
        .T
    )


def plot_total_production_diff_by_country(
    df_base: pd.DataFrame,
    df_scen: pd.DataFrame,
    *,
    base_name: str,
    scen_name: str,
    country_col: str = "Node",
    top_n: int | None = 23,
    sort_by_abs_total: bool = True,
    figsize=(16, 8),
    ylim=None,
):
    """
    Total production across all periods (TWh):
      - per country totals (sum over tech), printed as "(scen) - (base)"
      - per tech totals (sum over selected countries), printed as "(scen) - (base)"
      - plots diff stacked by tech per country
    """
    prod_base = compute_total_production_by_country_gwh(df_base, country_col=country_col)
    prod_scen = compute_total_production_by_country_gwh(df_scen, country_col=country_col)

    all_techs = sorted(set(prod_base.index) | set(prod_scen.index))
    all_ctry = sorted(set(prod_base.columns) | set(prod_scen.columns))

    prod_base = prod_base.reindex(index=all_techs, columns=all_ctry).fillna(0)
    prod_scen = prod_scen.reindex(index=all_techs, columns=all_ctry).fillna(0)

    base_twh = prod_base / 1e3
    scen_twh = prod_scen / 1e3
    diff_twh = (prod_scen - prod_base) / 1e3

    total_by_country = diff_twh.sum(axis=0)
    order = (total_by_country.abs() if sort_by_abs_total else total_by_country).sort_values(ascending=False).index
    if top_n is not None:
        order = order[:top_n]

    diff_twh = diff_twh[order]
    base_sel = base_twh[order]
    scen_sel = scen_twh[order]
    countries = list(diff_twh.columns)

    # Prints (levels as expressions, consistent with selected order)
    _print_series_expr(
        "Electricity: TOTAL production per country (sum over tech & periods) [TWh]",
        scen_sel.sum(axis=0), base_sel.sum(axis=0),
        unit=" TWh", decimals=2,
        scen_name=scen_name, base_name=base_name
    )
    _print_top_expr(
        "Electricity: tech with largest TOTAL impact (sum over selected countries) [TWh]",
        scen_sel.sum(axis=1), base_sel.sum(axis=1),
        n=20, unit=" TWh", decimals=2,
        scen_name=scen_name, base_name=base_name
    )

    # Plot
    x = np.arange(len(countries))
    fig, ax = plt.subplots(figsize=figsize)

    bottom_pos = np.zeros(len(countries))
    bottom_neg = np.zeros(len(countries))
    seen_labels = set()

    for tech in all_techs:
        vals = diff_twh.loc[tech].values
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
    ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=12)
    ax.set_ylabel("TOTAL production diff [TWh] (SCEN - BASE)", fontsize=12)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        y_min = min(0, bottom_neg.min()) * 1.1
        y_max = max(0, bottom_pos.max()) * 1.1
        ax.set_ylim([y_min if y_min != 0 else -1, y_max if y_max != 0 else 1])

    ax.grid(axis="y")
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left",
              title_fontsize=12, fontsize=10)
    plt.tight_layout()
    plt.show()

    return diff_twh


def plot_total_production_two_comparisons(
    df_base1: pd.DataFrame,
    df_scen1: pd.DataFrame,
    df_base2: pd.DataFrame,
    df_scen2: pd.DataFrame,
    *,
    base1_name: str,
    scen1_name: str,
    base2_name: str,
    scen2_name: str,
    label1: str,
    label2: str,
    width: float = 0.35,
):
    """
    Two separate scenario comparisons, printed as expressions and plotted as two stacked bars.

    IMPORTANT:
    This uses your original scaling: *5 (apparently years per investment period) and then sums over periods,
    so the output is "TWh across all periods" (i.e. not annual).
    """
    prod_base1, _ = compute_annual_production_gwh(df_base1)
    prod_scen1, _ = compute_annual_production_gwh(df_scen1)
    prod_base2, _ = compute_annual_production_gwh(df_base2)
    prod_scen2, _ = compute_annual_production_gwh(df_scen2)

    all_techs = sorted(set(prod_base1.index) | set(prod_scen1.index) | set(prod_base2.index) | set(prod_scen2.index))

    prod_base1 = prod_base1.reindex(all_techs).fillna(0)
    prod_scen1 = prod_scen1.reindex(all_techs).fillna(0)
    prod_base2 = prod_base2.reindex(all_techs).fillna(0)
    prod_scen2 = prod_scen2.reindex(all_techs).fillna(0)

    # Your original choice: convert annual GWh to "period-total" by multiplying 5,
    # then convert to TWh by /1e3 and sum across periods.
    base1_total_twh = (prod_base1 * 5 / 1e3).sum(axis=1)
    scen1_total_twh = (prod_scen1 * 5 / 1e3).sum(axis=1)
    base2_total_twh = (prod_base2 * 5 / 1e3).sum(axis=1)
    scen2_total_twh = (prod_scen2 * 5 / 1e3).sum(axis=1)

    # Print (levels as expressions)
    print("\n" + "=" * 80)
    print("Electricity: TOTAL production by technology across ALL periods [TWh] (two comparisons)")
    print(f"Bar 1 label: {label1}   -> comparison: {scen1_name} (SCEN) vs {base1_name} (BASE)")
    print(f"Bar 2 label: {label2}   -> comparison: {scen2_name} (SCEN) vs {base2_name} (BASE)")
    print("=" * 80)

    _print_top_expr(
        f"{label1}: total production by tech across all periods [TWh]",
        scen1_total_twh, base1_total_twh,
        n=25, unit=" TWh", decimals=2,
        scen_name=scen1_name, base_name=base1_name
    )
    _print_top_expr(
        f"{label2}: total production by tech across all periods [TWh]",
        scen2_total_twh, base2_total_twh,
        n=25, unit=" TWh", decimals=2,
        scen_name=scen2_name, base_name=base2_name
    )

    # Plot uses DIFF (SCEN - BASE) for each comparison as your original did
    diff1 = scen1_total_twh - base1_total_twh
    diff2 = scen2_total_twh - base2_total_twh

    eps = 1e-12
    techs_with_diff = [t for t in all_techs if abs(float(diff1.loc[t])) > eps or abs(float(diff2.loc[t])) > eps]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.array([0, 1])
    bottom_pos = np.zeros(2)
    bottom_neg = np.zeros(2)
    seen_labels = set()

    for tech in techs_with_diff:
        vals = np.array([float(diff1.loc[tech]), float(diff2.loc[tech])])
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
    ax.set_xticklabels([label1, label2], fontsize=12)
    ax.set_ylabel("TOTAL production diff [TWh] (SCEN - BASE)", fontsize=12)
    ax.grid(axis="y")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Technology", fontsize=9, title_fontsize=10,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"diff_1_TWh": diff1, "diff_2_TWh": diff2}, index=all_techs)


# =============================================================================
# ELECTRICITY: installed capacity
# =============================================================================
def compute_installed_capacity_mw(df: pd.DataFrame):
    """
    Returns:
      cap_mw: index=GeneratorType, columns=Period, values=MW installed
      periods: sorted periods
    """
    work = df.copy()
    work["genInstalledCap_MW"] = pd.to_numeric(work["genInstalledCap_MW"], errors="coerce").fillna(0)

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


def plot_installed_capacity_diff(df_base: pd.DataFrame, df_scen: pd.DataFrame, *, base_name: str, scen_name: str):
    """
    Installed capacity (GW) per period:
      - prints totals per period and totals per tech as "(scen) - (base)"
      - plots diff stacked by tech per period
    """
    cap_base_mw, periods = compute_installed_capacity_mw(df_base)
    cap_scen_mw, _ = compute_installed_capacity_mw(df_scen)

    all_techs = sorted(set(cap_base_mw.index) | set(cap_scen_mw.index))
    cap_base_mw = cap_base_mw.reindex(all_techs).fillna(0)
    cap_scen_mw = cap_scen_mw.reindex(all_techs).fillna(0)

    base_gw = cap_base_mw / 1e3
    scen_gw = cap_scen_mw / 1e3
    diff_gw = (cap_scen_mw - cap_base_mw) / 1e3

    # Prints
    _print_series_expr(
        "Electricity: installed capacity totals per period (sum over tech) [GW]",
        scen_gw.sum(axis=0), base_gw.sum(axis=0),
        unit=" GW", decimals=2,
        scen_name=scen_name, base_name=base_name
    )
    _print_top_expr(
        "Electricity: installed capacity totals per technology (sum over periods) [GW]",
        scen_gw.sum(axis=1), base_gw.sum(axis=1),
        n=25, unit=" GW", decimals=2,
        scen_name=scen_name, base_name=base_name
    )

    # Plot diff
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(16, 10))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    tol = 1e-12
    techs_with_diff = [t for t in all_techs if np.any(np.abs(diff_gw.loc[t].values) > tol)]

    for tech in techs_with_diff:
        vals = diff_gw.loc[tech].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        color = tech_colors.get(tech, "lightgray")

        ax.bar(x, pos, bottom=bottom_pos, color=color, edgecolor="black", linewidth=0.3, label=tech)
        bottom_pos += pos
        ax.bar(x, neg, bottom=bottom_neg, color=color, edgecolor="black", linewidth=0.3)
        bottom_neg += neg

    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Installed capacity diff [GW] (SCEN - BASE)", fontsize=12)
    ax.grid(axis="y")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    ax.legend(uniq_h, uniq_l, title="Technology",
              fontsize=9, title_fontsize=10,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return diff_gw


# =============================================================================
# HYDROGEN: annual production by tech (Mton/year) + storage capacity (Mton)
# =============================================================================
def plot_h2_annual_production_by_tech_diff(
    df_base: pd.DataFrame,
    df_scen: pd.DataFrame,
    *,
    base_name: str,
    scen_name: str,
    n_scen: int,
    n_hours: int,
    cols: tuple[str, str, str, str, str, str],
    y_lim=None,
):
    """
    Prints annual H2 production levels as "(scen) - (base)" in Mton/year:
      - totals per period (sum over tech)
      - totals per tech (sum over periods)
    Plots DIFF stacked bar by tech per period (Mton/year).
    """
    x1, x2, x3, x4, x5, x6 = cols
    season_scale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    def _prep(df):
        agg = (
            df.groupby("Period")
            .agg({x1: "sum", x2: "sum", x3: "sum", x4: "sum", x5: "sum", x6: "sum"})
            .reset_index()
        )
        agg["Period"] = pd.Categorical(agg["Period"], categories=PERIODS_ORDER, ordered=True)
        agg = agg.sort_values("Period").set_index("Period")
        agg[[x1, x2, x3, x4, x5, x6]] = agg[[x1, x2, x3, x4, x5, x6]] * season_scale / n_scen
        return agg.reindex(PERIODS_ORDER).fillna(0)

    base = _prep(df_base)
    scen = _prep(df_scen)
    diff = scen - base

    # Prints (levels)
    _print_series_expr(
        "Hydrogen: annual production totals per period (sum over tech) [Mton/year]",
        scen.sum(axis=1) / 1e6, base.sum(axis=1) / 1e6,
        unit=" Mton/yr", decimals=3,
        scen_name=scen_name, base_name=base_name
    )
    _print_top_expr(
        "Hydrogen: annual production totals per technology (sum over periods) [Mton/year]",
        scen.sum(axis=0) / 1e6, base.sum(axis=0) / 1e6,
        n=20, unit=" Mton/yr", decimals=3,
        scen_name=scen_name, base_name=base_name
    )

    # Plot diff (Mton)
    fig, ax = plt.subplots(figsize=(12, 8))
    pos_x = np.arange(len(PERIODS_ORDER)) * 0.5
    width = 0.3

    tech_cols = [x1, x2, x3, x4, x5, x6]
    labels = ["PEM green", "PEM yellow", "PEM import", "Alkaline", "SOEC", "Reformer"]
    colors = ["seagreen", "gold", "orange", "plum", "rebeccapurple", "grey"]

    bottom_pos = np.zeros(len(PERIODS_ORDER))
    bottom_neg = np.zeros(len(PERIODS_ORDER))

    for col, lab, color in zip(tech_cols, labels, colors):
        vals = diff[col].values / 1e6  # Mton/year
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
    ax.set_xticklabels(PERIODS_ORDER, rotation=0)
    ax.set_xlabel("Investment period", fontsize=12)
    ax.set_ylabel("Annual H2 production diff [Mton/year] (SCEN - BASE)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, title="Technology", title_fontsize=11)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    plt.tight_layout()
    plt.show()

    return diff


def plot_h2_storage_capacity_diff(
    df_base: pd.DataFrame,
    df_scen: pd.DataFrame,
    *,
    base_name: str,
    scen_name: str,
    periods=None,
):
    """
    Prints H2 storage capacity levels as "(scen) - (base)" in Mton:
      - totals per period (sum over storage types)
      - totals per storage type (sum over periods)
    Plots DIFF stacked bar by storage type per period (Mton).
    """
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
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
        return work.groupby("Period")[cols].sum().reindex(periods).fillna(0)

    base = _prep(df_base)
    scen = _prep(df_scen)
    diff_mton = (scen - base) / 1e6

    # Prints (levels)
    _print_series_expr(
        "Hydrogen: storage capacity totals per period (sum over storage types) [Mton]",
        scen.sum(axis=1) / 1e6, base.sum(axis=1) / 1e6,
        unit=" Mton", decimals=3,
        scen_name=scen_name, base_name=base_name
    )
    _print_top_expr(
        "Hydrogen: storage capacity totals per storage type (sum over periods) [Mton]",
        scen.sum(axis=0) / 1e6, base.sum(axis=0) / 1e6,
        n=10, unit=" Mton", decimals=3,
        scen_name=scen_name, base_name=base_name
    )

    # Plot diff
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(12, 7))

    bottom_pos = np.zeros(len(periods))
    bottom_neg = np.zeros(len(periods))

    for c, lab, col in zip(cols, labels, colors):
        vals = diff_mton[c].values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        ax.bar(x, pos, bottom=bottom_pos, color=col, edgecolor="black", linewidth=0.4, label=lab)
        bottom_pos += pos
        ax.bar(x, neg, bottom=bottom_neg, color=col, edgecolor="black", linewidth=0.4)
        bottom_neg += neg

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("H2 storage capacity diff [Mton] (SCEN - BASE)", fontsize=12)
    ax.set_xlabel("Period", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Storage type", fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.show()

    return diff_mton


# =============================================================================
# MAIN
# =============================================================================
def main():
    base_dir = SCENARIOS[BASE_NAME]
    scen_dir = SCENARIOS[SCEN_NAME]
    base2_dir = SCENARIOS[BASE2_NAME]
    scen2_dir = SCENARIOS[SCEN2_NAME]

    # -------------------------------------------------------------------------
    # OBJECTIVE COSTS
    # -------------------------------------------------------------------------
    _print_block_header("OBJECTIVE: cost decomposition", base_name=BASE_NAME, base_dir=base_dir, scen_name=SCEN_NAME, scen_dir=scen_dir)

    gen_op_s, gen_inv_s, h2_inv_s, stor_el_s, trans_inv_s, offconv_inv_s, obj_val_s, h2_op_s = load_objective_results(scen_dir)
    gen_op_b, gen_inv_b, h2_inv_b, stor_el_b, trans_inv_b, offconv_inv_b, obj_val_b, h2_op_b = load_objective_results(base_dir)

    df_scen_obj, obj_scen_bn = compute_period_costs(
        gen_op_s, gen_inv_s, h2_inv_s, stor_el_s, trans_inv_s, offconv_inv_s, obj_val_s, h2_op_s, N_SCEN
    )
    df_base_obj, obj_base_bn = compute_period_costs(
        gen_op_b, gen_inv_b, h2_inv_b, stor_el_b, trans_inv_b, offconv_inv_b, obj_val_b, h2_op_b, N_SCEN
    )

    plot_objective_single_bar(df_base_obj, obj_base_bn, title=f"Objective level (single scenario): {BASE_NAME}")
    plot_objective_single_bar(df_scen_obj, obj_scen_bn, title=f"Objective level (single scenario): {SCEN_NAME}")

    print("\n" + "-" * 80)
    print("Objective comparison summary (bn EUR)")
    print(f"  {SCEN_NAME}: {obj_scen_bn:.3f}")
    print(f"  {BASE_NAME}: {obj_base_bn:.3f}")
    print(f"  Diff (SCEN - BASE): {(obj_scen_bn - obj_base_bn):.3f}")

    plot_objective_components_diff(df_base_obj, df_scen_obj, base_name=BASE_NAME, scen_name=SCEN_NAME)
    _ = plot_total_objective_difference(
        df_base_obj, obj_base_bn, df_scen_obj, obj_scen_bn,
        base_name=BASE_NAME, scen_name=SCEN_NAME
    )

    # -------------------------------------------------------------------------
    # ELECTRICITY: generation investment file (annual production + installed cap)
    # -------------------------------------------------------------------------
    _print_block_header("ELECTRICITY: results_elec_generation_inv.csv", base_name=BASE_NAME, base_dir=base_dir, scen_name=SCEN_NAME, scen_dir=scen_dir)

    elec_base = load_elec_gen_inv(base_dir)
    elec_scen = load_elec_gen_inv(scen_dir)

    # Annual production (TWh/year) per period
    _ = plot_annual_production_diff(elec_base, elec_scen, base_name=BASE_NAME, scen_name=SCEN_NAME)

    # Total production by country across all periods (TWh)
    _ = plot_total_production_diff_by_country(
        elec_base, elec_scen,
        base_name=BASE_NAME, scen_name=SCEN_NAME,
        country_col="Node",
        top_n=23,
        sort_by_abs_total=True,
        figsize=(16, 8),
        ylim=None,
    )

    # Total production across all periods by tech (two comparisons)
    _print_block_header(
        "ELECTRICITY: total production across ALL periods (two comparisons)",
        base_name=BASE_NAME, base_dir=base_dir, scen_name=SCEN_NAME, scen_dir=scen_dir
    )
    print(f"Second comparison uses: BASE2={BASE2_NAME} ({base2_dir}) vs SCEN2={SCEN2_NAME} ({scen2_dir})")

    elec_base2 = load_elec_gen_inv(base2_dir)
    elec_scen2 = load_elec_gen_inv(scen2_dir)

    _ = plot_total_production_two_comparisons(
        df_base1=elec_base, df_scen1=elec_scen,
        df_base2=elec_base2, df_scen2=elec_scen2,
        base1_name=BASE_NAME, scen1_name=SCEN_NAME,
        base2_name=BASE2_NAME, scen2_name=SCEN2_NAME,
        label1=LABEL1, label2=LABEL2,
    )

    # Installed capacity (GW) per period
    _ = plot_installed_capacity_diff(elec_base, elec_scen, base_name=BASE_NAME, scen_name=SCEN_NAME)

    # -------------------------------------------------------------------------
    # HYDROGEN: production + storage
    # -------------------------------------------------------------------------
    _print_block_header("HYDROGEN: production + storage", base_name=BASE_NAME, base_dir=base_dir, scen_name=SCEN_NAME, scen_dir=scen_dir)

    h2_base = load_h2_prod(base_dir)
    h2_scen = load_h2_prod(scen_dir)

    _ = plot_h2_annual_production_by_tech_diff(
        h2_base, h2_scen,
        base_name=BASE_NAME, scen_name=SCEN_NAME,
        n_scen=N_SCEN, n_hours=12,
        cols=(
            "PEM_green production [ton]",
            "PEM_yellow production [ton]",
            "PEM_import production [ton]",
            "ALK production [ton]",
            "SOEC production [ton]",
            "Reformer production [ton]",
        ),
    )

    h2stor_base = load_h2_storage_inv(base_dir)
    h2stor_scen = load_h2_storage_inv(scen_dir)

    _ = plot_h2_storage_capacity_diff(
        h2stor_base, h2stor_scen,
        base_name=BASE_NAME, scen_name=SCEN_NAME
    )


if __name__ == "__main__":
    # Quick sanity checks: makes it obvious what is being compared
    for k, p in SCENARIOS.items():
        if not p.exists():
            raise FileNotFoundError(f"Scenario '{k}' dir does not exist: {p}")
    main()
