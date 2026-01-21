from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
project_dir = Path(__file__).resolve().parents[1]  # tilpass om du vil
data_dir = project_dir / "data"

# Fire resultatmapper totalt: 1 BASE + 3 SCENARIO som sammenlignes mot BASE
SCENARIOS = {
    "BASE": data_dir / "Results_FINAL_BASE_FLEX_emcap_cyclelim" / "full_model_base",
    "BFLAT": data_dir / "Results_FINAL_BASE_constantH2" / "full_model_base",
    "BSTORAGE": data_dir / "Results_woH2_storage" / "full_model_base",
    "BCONSTANT": data_dir / "Results_FINAL_BASE_NOFLEX_emcap_cyclelim" / "full_model_base",
}


BASE_KEY = "BASE"
COMPARE_KEYS = ["BFLAT", "BSTORAGE", "BCONSTANT"]  # disse får hver sin søyle

# Skaleringsparametre (samme idé som i koden din)
N_SCEN = 2          # antall stokastiske scenarier (for operational cost / H2 annualisering)
N_HOURS = 12        # brukt i season_scale i H2-annualisering
YEARS_PER_PERIOD = 5  # samme som i din "total electricity across all periods" (*5)

# Period ordering (hvis du vil bruke den til noe senere)
PERIODS_ORDER = [
    "2020-2025", "2025-2030", "2030-2035", "2035-2040",
    "2040-2045", "2045-2050", "2050-2055",
]

# =============================================================================
# COLORS (lånt/tilpasset fra din kode)
# =============================================================================
OBJ_COLORS_11 = [
    "teal", "darkturquoise", "orange", "yellowgreen", "seagreen",
    "plum", "moccasin", "hotpink", "darkslategrey", "brown", "dodgerblue",
]
OBJ_COLORS_12 = OBJ_COLORS_11 + ["darkgrey"]  # + Other

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

H2_TECH_COLS = (
    "PEM_green production [ton]",
    "PEM_yellow production [ton]",
    "PEM_import production [ton]",
    "ALK production [ton]",
    "SOEC production [ton]",
    "Reformer production [ton]",
)
H2_TECH_LABELS = ["PEM green", "PEM yellow", "PEM import", "Alkaline", "SOEC", "Reformer"]
H2_TECH_COLORS = {
    "PEM green": "seagreen",
    "PEM yellow": "gold",
    "PEM import": "orange",
    "Alkaline": "plum",
    "SOEC": "rebeccapurple",
    "Reformer": "grey",
}

H2_STORAGE_COLS = [
    "Total SaltCavern storage capacity [ton]",
    "Total DGF storage capacity [ton]",
    "Total Aquifer storage capacity [ton]",
]
H2_STORAGE_LABELS = ["Salt cavern", "DGF", "Aquifer"]
H2_STORAGE_COLORS = {
    "Salt cavern": "lightskyblue",
    "DGF": "orange",
    "Aquifer": "yellowgreen",
}


# =============================================================================
# HELPERS
# =============================================================================
def _require_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _bn(x: pd.Series | float) -> pd.Series | float:
    return x / 1e9


def _period_sum(df: pd.DataFrame, col: str, denom: float = 1.0) -> pd.Series:
    return df.groupby("Period")[col].sum() / denom


def _cum_to_periodic(cum_series: pd.Series) -> pd.Series:
    cum_series = cum_series.sort_index()
    return cum_series.diff().fillna(cum_series)


def _stacked_multi_bar_plot(
    diffs: dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    color_for: dict[str, str] | None = None,
    default_color: str = "lightgray",
    order_by_abs_total: bool = True,
    figsize=(11, 7),
    width: float = 0.28,
):
    """
    diffs: mapping bar_label -> Series(category -> value)
    Plots 3 bars (one per label in diffs) with stacked positive and negative contributions.
    """
    bar_labels = list(diffs.keys())

    # Union categories
    cats = set()
    for s in diffs.values():
        cats |= set(pd.Series(s).index)
    cats = list(cats)

    # Build aligned matrix [n_cats x n_bars]
    mat = pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in diffs.items()}).reindex(cats).fillna(0.0)

    if order_by_abs_total:
        order = mat.abs().sum(axis=1).sort_values(ascending=False).index.tolist()
        mat = mat.loc[order]

    n_bars = len(bar_labels)
    x = np.arange(n_bars)

    fig, ax = plt.subplots(figsize=figsize)
    bottom_pos = np.zeros(n_bars)
    bottom_neg = np.zeros(n_bars)

    seen = set()
    eps = 1e-12

    for cat in mat.index:
        vals = mat.loc[cat].values.astype(float)
        if np.all(np.abs(vals) < eps):
            continue

        col = (color_for or {}).get(cat, default_color)

        for i in range(n_bars):
            v = float(vals[i])
            if abs(v) < eps:
                continue

            lbl = cat if cat not in seen else None
            bottom = bottom_pos[i] if v > 0 else bottom_neg[i]
            ax.bar(x[i], v, width=width, bottom=bottom, color=col,
                   edgecolor="black", linewidth=0.3, label=lbl)

            if v > 0:
                bottom_pos[i] += v
            else:
                bottom_neg[i] += v

            if lbl is not None:
                seen.add(cat)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_ylabel(ylabel, fontsize=20)
    ymin = min(bottom_neg) * 1.1
    ymax = max(bottom_pos) * 1.1
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=16)

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd

def pct_change(scen: float, base: float) -> float:
    if base == 0:
        return np.nan
    return 100.0 * (scen - base) / base

def make_summary_table(base_s: pd.Series, scen_s: pd.Series, unit: str, add_shares: bool = True) -> pd.DataFrame:
    base = base_s.reindex(scen_s.index).fillna(0.0)
    scen = scen_s.fillna(0.0)

    total_base = float(base.sum())
    total_scen = float(scen.sum())
    share = (scen / total_scen) if (add_shares and total_scen != 0) else None

    rows = []
    # Total row
    rows.append({
        "Item": "TOTAL",
        "Base": total_base,
        "Scenario": total_scen,
        "Abs diff": total_scen - total_base,
        "% diff vs Base": pct_change(total_scen, total_base),
        "Scenario share": 1.0 if share is not None else np.nan,
        "Unit": unit,
    })

    # Breakdown rows
    for k in scen.index:
        rows.append({
            "Item": k,
            "Base": float(base[k]),
            "Scenario": float(scen[k]),
            "Abs diff": float(scen[k] - base[k]),
            "% diff vs Base": pct_change(float(scen[k]), float(base[k])),
            "Scenario share": float(share[k]) if share is not None else np.nan,
            "Unit": unit,
        })

    return pd.DataFrame(rows)


# =============================================================================
# LOADERS (samme filer som i din kode)
# =============================================================================
def load_objective_results(result_dir: Path):
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

def load_h2_reformer_detailed_inv(result_dir: Path) -> pd.DataFrame:
    return pd.read_csv(result_dir / "results_hydrogen_reformer_detailed_inv.csv")


def total_reformer_net_electricity_twh(result_dir: Path, years_per_period: int = 5) -> float:
    df = load_h2_reformer_detailed_inv(result_dir)

    col = "Expected electricity consumption [GWh]"
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Net contribution to electricity balance (positive = supply, negative = demand)
    df["Reformer net electricity [GWh]"] = -df[col]

    by_period = df.groupby("Period")["Reformer net electricity [GWh]"].sum()  # GWh/year per period
    total_twh = float((by_period * years_per_period / 1e3).sum())            # TWh across all periods
    return total_twh


# =============================================================================
# OBJECTIVE: compute diffs (bar = SCEN - BASE)
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
    # Generator operational (discounted; scale by stochastic scenarios)
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

    # Repurposed H2 pipeline + CO2 pipeline (already periodic)
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


def objective_component_diff_series(
    *,
    base_dir: Path,
    scen_dir: Path,
    n_scen: int,
) -> pd.Series:
    # Load
    gen_op_s, gen_inv_s, h2_inv_s, stor_el_s, trans_inv_s, offconv_inv_s, obj_val_s, h2_op_s = load_objective_results(scen_dir)
    gen_op_b, gen_inv_b, h2_inv_b, stor_el_b, trans_inv_b, offconv_inv_b, obj_val_b, h2_op_b = load_objective_results(base_dir)

    # Compute
    df_scen, obj_scen_bn = compute_period_costs(
        gen_op_s, gen_inv_s, h2_inv_s, stor_el_s, trans_inv_s, offconv_inv_s, obj_val_s, h2_op_s, n_scen
    )
    df_base, obj_base_bn = compute_period_costs(
        gen_op_b, gen_inv_b, h2_inv_b, stor_el_b, trans_inv_b, offconv_inv_b, obj_val_b, h2_op_b, n_scen
    )

    # Total component diffs (sum over periods)
    comp_diff = (df_scen - df_base).sum(axis=0)  # bn EUR
    obj_diff = obj_scen_bn - obj_base_bn
    other_diff = obj_diff - comp_diff.sum()

    out = comp_diff.copy()
    out["Other"] = other_diff
    return out

def objective_component_level_series(*, result_dir: Path, n_scen: int) -> pd.Series:
    gen_op, gen_inv, h2_inv, stor_el, trans_inv, offconv_inv, obj_val, h2_op = load_objective_results(result_dir)
    df_period, obj_bn = compute_period_costs(
        gen_op, gen_inv, h2_inv, stor_el, trans_inv, offconv_inv, obj_val, h2_op, n_scen
    )

    comp_total = df_period.sum(axis=0)               # bn EUR per component
    other = obj_bn - float(comp_total.sum())
    out = comp_total.copy()
    out["Other"] = other
    return out

# =============================================================================
# ELECTRICITY: total production by tech across all periods (TWh)
# =============================================================================
def compute_annual_production_gwh(df: pd.DataFrame) -> pd.DataFrame:
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
    return prod[periods]


def total_electricity_by_tech_twh(df: pd.DataFrame, years_per_period: int = 5) -> pd.Series:
    # annual GWh/year per period -> multiply by years_per_period -> GWh across all periods -> /1e3 -> TWh
    prod_gwh = compute_annual_production_gwh(df)
    total_twh = (prod_gwh * years_per_period / 1e3).sum(axis=1)
    return total_twh


# =============================================================================
# HYDROGEN: total production by tech across all periods (Mton)
# =============================================================================
def total_h2_production_by_tech_mton(
    df: pd.DataFrame,
    *,
    n_scen: int,
    n_hours: int,
    years_per_period: int = 5,
    cols: tuple[str, ...] = H2_TECH_COLS,
    labels: list[str] = H2_TECH_LABELS,
) -> pd.Series:
    # Samme annualisering som hos deg
    season_scale = (8760 - 2 * n_hours) / (4 * 7 * n_hours)

    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    by_period = work.groupby("Period")[list(cols)].sum()

    # Annual ton/year per period (for hver tech)
    annual_ton_per_period = by_period * season_scale / n_scen

    # Total over alle perioder: *years_per_period og summer
    total_ton_all_periods = (annual_ton_per_period * years_per_period).sum(axis=0)

    # Map til tech-labels og Mton
    s = pd.Series(total_ton_all_periods.values, index=labels, dtype=float) / 1e6
    return s


# =============================================================================
# HYDROGEN: total storage capacity by type (Mton)
# =============================================================================
def total_h2_storage_capacity_by_type_mton(
    df: pd.DataFrame,
    *,
    cols: list[str] = H2_STORAGE_COLS,
    labels: list[str] = H2_STORAGE_LABELS,
) -> pd.Series:
    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    by_period = work.groupby("Period")[cols].sum()

    # "Total" her: sum over perioder (samme logikk som din "totals per storage type (sum over periods)")
    total_ton = by_period.loc["2050-2055"]
    s = pd.Series(total_ton.values, index=labels, dtype=float) / 1e6
    return s


# =============================================================================
# PLOTS (3 bars: SCEN_A/B/C vs BASE)
# =============================================================================
def plot_objective_components_3vs1(base_dir: Path, scen_dirs: dict[str, Path]):
    diffs = {}
    tables = {}

    # nivå-serie for base (bn EUR per komponent + Other)
    base_level = objective_component_level_series(result_dir=base_dir, n_scen=N_SCEN)

    for scen_key, scen_dir in scen_dirs.items():
        # nivå-serie for scenario
        scen_level = objective_component_level_series(result_dir=scen_dir, n_scen=N_SCEN)

        # diff til plottet
        diffs[scen_key] = (scen_level - base_level)

        # tabell med nivå + diff + % (her gir shares lite mening -> add_shares=False)
        tables[scen_key] = make_summary_table(base_level, scen_level, unit="bn EUR", add_shares=False)

        print(f"\n=== Objective components: {scen_key} vs BASE ===")
        print(tables[scen_key].to_string(index=False))

    # farger (basert på alle kategorier som faktisk finnes)
    all_cats = sorted(set().union(*[set(s.index) for s in diffs.values()]))
    color_map = {c: OBJ_COLORS_12[i % len(OBJ_COLORS_12)] for i, c in enumerate(all_cats)}

    _stacked_multi_bar_plot(
        diffs,
        title="Objective components (TOTAL) – difference (SCEN - BASE)",
        ylabel="Objective value [bn EUR]",
        color_for=color_map,
        figsize=(11, 7),
        width=0.28,
    )

    return tables  # valgfritt


def plot_total_electricity_production_3vs1(base_dir: Path, scen_dirs: dict[str, Path]):
    base_df = load_elec_gen_inv(base_dir)
    base_total = total_electricity_by_tech_twh(base_df, years_per_period=YEARS_PER_PERIOD)

    # Add reformer net electricity as its own "tech"
    base_total.loc["Reformer (net el)"] = total_reformer_net_electricity_twh(
        base_dir, years_per_period=YEARS_PER_PERIOD
    )

    diffs = {}
    tables = {}

    for scen_key, scen_dir in scen_dirs.items():
        scen_df = load_elec_gen_inv(scen_dir)
        scen_total = total_electricity_by_tech_twh(scen_df, years_per_period=YEARS_PER_PERIOD)

        scen_total.loc["Reformer (net el)"] = total_reformer_net_electricity_twh(
            scen_dir, years_per_period=YEARS_PER_PERIOD
        )

        diffs[scen_key] = (scen_total - base_total)
        tables[scen_key] = make_summary_table(base_total, scen_total, unit="TWh", add_shares=True)

        print(f"\n=== Electricity production summary: {scen_key} vs BASE ===")
        print(tables[scen_key].to_string(index=False))

    # Make sure it has a color
    tech_colors_with_reformer = dict(tech_colors)
    tech_colors_with_reformer["Reformer (net el)"] = "grey"

    _stacked_multi_bar_plot(
        diffs,
        title=f"Electricity production (TOTAL across all periods) – diff (SCEN - BASE) [*{YEARS_PER_PERIOD} years/period]",
        ylabel="Total expected electricity production [TWh]",
        color_for=tech_colors_with_reformer,
        default_color="lightgray",
        figsize=(12, 7),
        width=0.28,
    )

    return tables


def plot_total_h2_production_3vs1(base_dir: Path, scen_dirs: dict[str, Path]):
    base_df = load_h2_prod(base_dir)
    base_total = total_h2_production_by_tech_mton(
        base_df,
        n_scen=N_SCEN,
        n_hours=N_HOURS,
        years_per_period=YEARS_PER_PERIOD,
    )

    diffs = {}
    tables = {}  # <-- NYTT

    for scen_key, scen_dir in scen_dirs.items():
        scen_df = load_h2_prod(scen_dir)
        scen_total = total_h2_production_by_tech_mton(
            scen_df,
            n_scen=N_SCEN,
            n_hours=N_HOURS,
            years_per_period=YEARS_PER_PERIOD,
        )

        diffs[scen_key] = (scen_total - base_total)

        # <-- NYTT: tabell med nivå + diff + % + andeler (share)
        tables[scen_key] = make_summary_table(base_total, scen_total, unit="Mton", add_shares=True)

        # <-- NYTT: print (valgfritt)
        print(f"\n=== H2 production summary: {scen_key} vs BASE ===")
        print(tables[scen_key].to_string(index=False))

    _stacked_multi_bar_plot(
        diffs,
        title=f"Hydrogen production (TOTAL across all periods) – diff (SCEN - BASE) [annualized, *{YEARS_PER_PERIOD}]",
        ylabel="Total hydrogen production [Mton]",
        color_for=H2_TECH_COLORS,
        default_color="lightgray",
        figsize=(11, 7),
        width=0.28,
    )

    return tables  # <-- valgfritt



def plot_total_h2_storage_capacity_3vs1(base_dir: Path, scen_dirs: dict[str, Path]):
    base_df = load_h2_storage_inv(base_dir)
    base_total = total_h2_storage_capacity_by_type_mton(base_df)

    diffs = {}
    tables = {}  # <-- NYTT

    for scen_key, scen_dir in scen_dirs.items():
        scen_df = load_h2_storage_inv(scen_dir)
        scen_total = total_h2_storage_capacity_by_type_mton(scen_df)

        diffs[scen_key] = (scen_total - base_total)

        # <-- NYTT: tabell med nivå + diff + % + andeler
        tables[scen_key] = make_summary_table(base_total, scen_total, unit="Mton", add_shares=True)

        # <-- NYTT: print (valgfritt)
        print(f"\n=== H2 storage capacity summary: {scen_key} vs BASE ===")
        print(tables[scen_key].to_string(index=False))

    _stacked_multi_bar_plot(
        diffs,
        title="Hydrogen storage capacity – diff (SCEN - BASE)",  # evt spesifiser "final period" hvis det er det du bruker
        ylabel="Δ UHS capacity (SCEN - BASE) [Mton]",           # mer korrekt enn "Total" når du plottet diff
        color_for=H2_STORAGE_COLORS,
        default_color="lightgray",
        figsize=(11, 7),
        width=0.28,
    )

    return tables  # <-- valgfritt



# =============================================================================
# MAIN
# =============================================================================
def main():
    # Sanity checks
    for k, p in SCENARIOS.items():
        if not p.exists():
            raise FileNotFoundError(f"Scenario '{k}' dir does not exist: {p}")

    base_dir = SCENARIOS[BASE_KEY]
    scen_dirs = {k: SCENARIOS[k] for k in COMPARE_KEYS}

    # 1) Objective components (TOTAL)
    plot_objective_components_3vs1(base_dir, scen_dirs)

    # 2) Total electricity production by tech (TOTAL across periods)
    plot_total_electricity_production_3vs1(base_dir, scen_dirs)

    # 3) Total hydrogen production by tech (TOTAL across periods)
    plot_total_h2_production_3vs1(base_dir, scen_dirs)

    # 4) Total hydrogen storage capacity by type (TOTAL sum over periods)
    plot_total_h2_storage_capacity_3vs1(base_dir, scen_dirs)



if __name__ == "__main__":
    main()
