from pathlib import Path
from typing import Optional, Dict, Tuple, Union, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# -----------------------------
# Felles: finn + last result-csv
# -----------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines


def _period_key(p: str):
    # robust nok for "2020-2025"
    a, b = str(p).split("-")
    return (int(b), int(a))  # sorter på sluttår først


def hydrogen_pipeline_capacity_map(
    df: pd.DataFrame,
    *,
    capacity_col: str = "Pipeline total capacity [ton/hr]",
    period: str | None = None,          # None => siste periode
    color: str = "tab:blue",
    agg: str = "max",                    # "max" anbefales for å unngå dobbelttelling
    min_cap: float = 1e-12,
    title: str | None = None,
    savefigure: bool = False,
    figurename: str | None = None,
    results_dir: str | Path | None = None,
):
    d = df.copy()

    required = {"Between node", "And node", "Period", capacity_col}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Mangler kolonner: {missing}")

    # Velg siste periode hvis ikke oppgitt
    if period is None:
        periods = sorted(d["Period"].astype(str).unique(), key=_period_key)
        if not periods:
            raise ValueError("Fant ingen Period-verdier.")
        period = periods[-1]

    d = d[d["Period"].astype(str) == str(period)].copy()
    if d.empty:
        raise ValueError(f"Ingen rader for Period='{period}'")

    # Retningsuavhengig nodepar
    d["node_pair"] = d.apply(
        lambda r: tuple(sorted([str(r["Between node"]), str(r["And node"])])),
        axis=1
    )

    # Aggreger kapasitet per par (max hindrer dobbelttelling hvis begge retninger finnes)
    if agg == "max":
        df_sum = d.groupby("node_pair", as_index=False)[capacity_col].max()
    elif agg == "sum":
        df_sum = d.groupby("node_pair", as_index=False)[capacity_col].sum()
    else:
        raise ValueError("agg må være 'max' eller 'sum'")

    df_sum = df_sum[df_sum[capacity_col] > float(min_cap)]
    if df_sum.empty:
        raise ValueError(f"Ingen kapasitet > {min_cap} i perioden {period}")

    # Koordinater (samme som du bruker)
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

    # Bygg linjer
    lines = []
    for _, row in df_sum.iterrows():
        n1, n2 = row["node_pair"]
        c1 = node_coords.get(n1)
        c2 = node_coords.get(n2)
        if c1 and c2:
            lines.append({"coords": [c1, c2], "value": float(row[capacity_col]), "nodes": f"{n1}–{n2}"})

    if not lines:
        raise ValueError("Ingen linjer å plotte (mangler koordinater for alle par).")

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-9, 30, 34, 72], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)

    max_val = max(l["value"] for l in lines)

    # Diskrete nivåer (som din stil)
    bins = [0.0, 0.05,0.2, 0.4, 0.6,0.8, 1.0]
    widths = [1,6, 11, 16,24, 30]

    for l in lines:
        ratio = l["value"] / max_val if max_val > 0 else 0.0
        lw = widths[-1]
        for i in range(len(bins) - 1):
            if bins[i] <= ratio < bins[i + 1]:
                lw = widths[i]
                break
        xs, ys = zip(*l["coords"])
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.55, transform=ccrs.PlateCarree())

    # Legend
    legend_lines = []
    min_pos = df_sum.loc[df_sum[capacity_col] > 0, capacity_col].min()
    for i in range(len(widths)):
        lo = bins[i] * max_val
        if lo == 0:
            lo = float(min_pos)
        hi = bins[i + 1] * max_val
        legend_lines.append(
            mlines.Line2D([], [], color=color, linewidth=widths[i], label=f"{lo:.2f} – {hi:.2f} ton/hr")
        )

    ax.legend(
        handles=legend_lines,
        title=f"{capacity_col}",
        loc="upper left",
        frameon=True,
        labelspacing=1.2,
        fontsize=20,
        title_fontsize=20,
    )

    # Noder
    for name, (lon, lat) in node_coords.items():
        ax.plot(lon, lat, marker="o", color="black", markersize=3, transform=ccrs.PlateCarree())
    fig.tight_layout()

    if savefigure and results_dir and figurename:
        outdir = Path(results_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        figpath = outdir / f"{figurename}_hydrogenPipelineMap_{period}.png"
        plt.savefig(figpath, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {figpath}")

    plt.show()
    return df_sum, fig, ax


def find_result_files(base_dir: Union[str, Path], filename: str) -> list[Path]:
    base = Path(base_dir).expanduser().resolve()
    return sorted(base.rglob(filename))


def load_link_operational(
    base_dir: Union[str, Path],
    *,
    filename: str,
    from_col: str,
    to_col: str,
    value_col: str,
    period_col: str = "Period",
    season_col: str = "Season",
    scenario_col: str = "Scenario",
    gas_scenario_col: str = "GasScenario",
    hour_col: str = "Hour",
    extra_numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Leser alle matchende csv-filer rekursivt og standardiserer kolonnenavn til:
      FromNode, ToNode, Period, Season, Scenario, GasScenario, Hour, Value (+ evt. ekstra)
    """
    files = find_result_files(base_dir, filename)
    if not files:
        raise FileNotFoundError(f"Fant ingen '{filename}' under {Path(base_dir).resolve()}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["__run__"] = f.parent.name
        df["__path__"] = str(f)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # rename til standard
    rename_map = {
        from_col: "FromNode",
        to_col: "ToNode",
        period_col: "Period",
        season_col: "Season",
        scenario_col: "Scenario",
        gas_scenario_col: "GasScenario",
        hour_col: "Hour",
        value_col: "Value",
    }
    out = out.rename(columns=rename_map)

    # numerikk
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

    if extra_numeric_cols:
        for c in extra_numeric_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

    # dropp rader uten noder/verdi
    out = out.dropna(subset=["FromNode", "ToNode", "Value"])

    # period som str for robust matching (f.eks. '2050-2055')
    out["Period"] = out["Period"].astype(str).str.strip()

    # også noder som str
    out["FromNode"] = out["FromNode"].astype(str)
    out["ToNode"] = out["ToNode"].astype(str)

    return out


# -----------------------------
# Vekter (scenario/gass) + aggregasjon
# -----------------------------

def _normalize_weights_from_unique(values: Iterable, provided: Optional[Dict] = None) -> Dict:
    uniq = sorted(set(values))
    if not uniq:
        return {}
    if provided is None:
        p = 1.0 / len(uniq)
        return {u: p for u in uniq}
    w = {u: float(provided.get(u, 0.0)) for u in uniq}
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w


def link_activity_matrix(
    df: pd.DataFrame,
    period: Union[int, str],
    *,
    run: Optional[str] = None,
    mode: str = "expected_total",   # se under
    season_scale: Optional[Dict[str, float]] = None,
    scenario_prob: Optional[Dict] = None,
    gas_scenario_prob: Optional[Dict] = None,
    nodes: Optional[list[str]] = None,
    # enhet/konvertering
    value_is: str = "MW",            # "MW" eller "H2_ton_per_h" (for labeling/konvertering)
    h2_mwh_per_ton: float = 33.3,    # brukes hvis value_is == "H2_ton_per_h" og du vil til TWh
) -> Tuple[pd.DataFrame, str]:

    if df.empty:
        raise ValueError("Input-dataframe er tom.")

    d = df.copy()

    if run is not None:
        d = d[d["__run__"].astype(str) == str(run)]
        if d.empty:
            raise ValueError(f"Ingen rader etter run-filter: run='{run}'")

    d = d[d["Period"] == str(period)]
    if d.empty:
        raise ValueError(f"Ingen rader for Period='{period}'")

    if mode == "mean":
        agg = d.groupby(["FromNode", "ToNode"], as_index=False)["Value"].mean()
        unit = value_is
        agg = agg.rename(columns={"Value": "value"})

    elif mode == "sum":
        agg = d.groupby(["FromNode", "ToNode"], as_index=False)["Value"].sum()
        unit = value_is
        agg = agg.rename(columns={"Value": "value"})

    elif mode in ("expected_total", "expected_TWh"):
        season_scale = season_scale or {}

        if "Season" in d.columns:
            d["__seas__"] = d["Season"].astype(str)
            d["__seas_scale__"] = d["__seas__"].map(season_scale).fillna(1.0)
        else:
            d["__seas_scale__"] = 1.0

        if "Scenario" in d.columns:
            sp = _normalize_weights_from_unique(d["Scenario"], scenario_prob)
            d["__p_s__"] = d["Scenario"].map(sp).fillna(0.0)
        else:
            d["__p_s__"] = 1.0

        if "GasScenario" in d.columns:
            gp = _normalize_weights_from_unique(d["GasScenario"], gas_scenario_prob)
            d["__p_g__"] = d["GasScenario"].map(gp).fillna(0.0)
        else:
            d["__p_g__"] = 1.0

        # forventet total: Value * timer * sannsynligheter
        d["__wtotal__"] = d["Value"] * d["__seas_scale__"] * d["__p_s__"] * d["__p_g__"]
        agg = d.groupby(["FromNode", "ToNode"], as_index=False)["__wtotal__"].sum()
        agg = agg.rename(columns={"__wtotal__": "value"})

        if mode == "expected_total":
            if value_is == "MW":
                unit = "MWh (expected)"
            elif value_is == "H2_ton_per_h":
                unit = "ton H2 (expected)"
            else:
                unit = "expected total"
        else:
            # expected_TWh
            if value_is == "MW":
                agg["value"] = agg["value"] / 1e6  # MWh -> TWh
                unit = "TWh (expected)"
            elif value_is == "H2_ton_per_h":
                agg["value"] = (agg["value"] * h2_mwh_per_ton) / 1e6  # ton -> MWh -> TWh
                unit = "TWh_H2_LHV (expected)"
            else:
                raise ValueError("Ukjent value_is for expected_TWh.")
    else:
        raise ValueError("mode må være en av: 'mean', 'sum', 'expected_total', 'expected_TWh'")

    # node-rekkefølge
    if nodes is None:
        nodes = sorted(set(agg["FromNode"]).union(set(agg["ToNode"])))
    else:
        nodes = [str(n) for n in nodes]

    mat = (
        agg.pivot_table(index="FromNode", columns="ToNode", values="value", aggfunc="sum", fill_value=0.0)
           .reindex(index=nodes, columns=nodes, fill_value=0.0)
    )
    return mat, unit


def drop_all_zero_rows_cols(mat: pd.DataFrame, tol: float = 0.0) -> pd.DataFrame:
    a = mat.to_numpy()
    if tol > 0:
        keep_r = (np.abs(a) > tol).any(axis=1)
        keep_c = (np.abs(a) > tol).any(axis=0)
    else:
        keep_r = (a != 0).any(axis=1)
        keep_c = (a != 0).any(axis=0)
    return mat.loc[keep_r, keep_c]


# -----------------------------
# Plot: (bedre lesbarhet) grid + tall (terskel)
# -----------------------------

def plot_diff_heatmap(
    mat: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    figsize_scale: float = 1.3,
    dpi: int = 350,
    show_tick_every: Optional[int] = None,
    grid: bool = True,
    annotate: bool = True,
    annotate_fmt: str = ".2f",
    annotate_min_abs: Optional[float] = None,
    max_annotate_n: int = 70,
    savepath: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    n = mat.shape[0]
    vals = mat.to_numpy()

    vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    if vmax == 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    w = max(14.0, n * 0.45) * figsize_scale
    h = max(12.0, n * 0.38) * figsize_scale

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel("Import node",fontsize=12)
    ax.set_ylabel("Export node",fontsize=12)

    if show_tick_every is None:
        show_tick_every = max(1, n // 45)

    xt = np.arange(0, n, show_tick_every)
    yt = np.arange(0, n, show_tick_every)
    ax.set_xticks(xt)
    ax.set_yticks(yt)

    ax.set_xticklabels([mat.columns[i] for i in xt], rotation=90, fontsize=12)
    ax.set_yticklabels([mat.index[i] for i in yt], fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if grid:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

    if annotate and n <= max_annotate_n:
        if annotate_min_abs is None:
            annotate_min_abs = 0.05 * vmax
        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                if abs(v) >= annotate_min_abs:
                    ax.text(j, i, format(v, annotate_fmt), ha="center", va="center", fontsize=12)

    fig.tight_layout()
    if savepath is not None:
        savepath = Path(savepath).expanduser().resolve()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    return fig, ax


# -----------------------------
# Differanse (dir2 - dir1) for valgfri link-type
# -----------------------------

def align_square(m1: pd.DataFrame, m2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = sorted(set(m1.index).union(m1.columns).union(m2.index).union(m2.columns))
    return (
        m1.reindex(index=nodes, columns=nodes, fill_value=0.0),
        m2.reindex(index=nodes, columns=nodes, fill_value=0.0),
    )


def top_nodes_by_abs(mat_diff: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
    if top_k is None:
        return mat_diff
    n = mat_diff.shape[0]
    if top_k <= 0 or top_k >= n:
        return mat_diff
    a = np.abs(mat_diff.to_numpy())
    score = a.sum(axis=1) + a.sum(axis=0)
    idx = np.argsort(score)[::-1][:top_k]
    nodes = mat_diff.index.to_numpy()[idx].tolist()
    return mat_diff.loc[nodes, nodes]


def diff_matrix_from_dirs(
    result_dir1: Union[str, Path],
    result_dir2: Union[str, Path],
    *,
    loader_kwargs: Dict,
    period: Union[int, str],
    run: Optional[str] = None,
    mode: str = "expected_total",
    season_scale: Optional[Dict[str, float]] = None,
    scenario_prob: Optional[Dict] = None,
    gas_scenario_prob: Optional[Dict] = None,
    value_is: str = "MW",
    h2_mwh_per_ton: float = 33.3,
    drop_zeros: bool = True,
    zero_tol: float = 0.0,
    top_k: Optional[int] = 60,
) -> Tuple[pd.DataFrame, str]:
    df1 = load_link_operational(result_dir1, **loader_kwargs)
    df2 = load_link_operational(result_dir2, **loader_kwargs)

    m1, unit = link_activity_matrix(
        df1, period,
        run=run, mode=mode,
        season_scale=season_scale,
        scenario_prob=scenario_prob,
        gas_scenario_prob=gas_scenario_prob,
        value_is=value_is,
        h2_mwh_per_ton=h2_mwh_per_ton,
    )
    m2, _ = link_activity_matrix(
        df2, period,
        run=run, mode=mode,
        season_scale=season_scale,
        scenario_prob=scenario_prob,
        gas_scenario_prob=gas_scenario_prob,
        value_is=value_is,
        h2_mwh_per_ton=h2_mwh_per_ton,
    )

    m1, m2 = align_square(m1, m2)
    diff = m1 - m2

    if drop_zeros:
        diff = drop_all_zero_rows_cols(diff, tol=zero_tol)

    diff = top_nodes_by_abs(diff, top_k=top_k)
    return diff, unit

ELEC_LOADER = dict(
    filename="results_elec_transmission_operational.csv",
    from_col="FromNode",
    to_col="ToNode",
    value_col="TransmissionReceived_MW",
    extra_numeric_cols=["Losses_MW"],
)

H2_PIPE_LOADER = dict(
    filename="results_hydrogen_pipeline_operational.csv",
    from_col="From node",
    to_col="To node",
    value_col="Hydrogen sent [ton]",
)

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"

result_dir1 = data_dir / "Results_FINAL_BASE_NOFLEX_emcap_cyclelim" / "full_model_base"
result_dir2 = data_dir / "Results_FINAL_BASE_FLEX_emcap_cyclelim" / "full_model_base"

seasScale = {
    "winter": 26.0, "spring": 26.0, "summer": 26.0, "fall": 26.0,
    "peak1": 12.0, "peak2": 12.0,
}

diff_h2, unit_h2 = diff_matrix_from_dirs(
    result_dir1, result_dir2,
    loader_kwargs=H2_PIPE_LOADER,
    period="2050-2055",
    mode="expected_TWh",
    season_scale=seasScale,
    value_is="H2_ton_per_h",
    h2_mwh_per_ton=33.3,
    top_k=15,
    zero_tol=1e-9,
)

fig, ax = plot_diff_heatmap(
    diff_h2,
    title="H2 pipeline export difference 2050-2055",
    cbar_label=f"Difference [ton] BCONSTANT - BFLEX",
    grid=True,
    annotate=True,
    annotate_min_abs=None,
    dpi=400,
    savepath="out/h2_pipe_diff_2050-2055.png",
)

plt.show()

df_h2 = pd.read_csv(result_dir1/"results_hydrogen_pipeline_inv.csv")

df_sum_last, fig, ax = hydrogen_pipeline_capacity_map(
    df_h2,
    capacity_col="Pipeline total capacity [ton/hr]",  # evt "Pipeline capacity built [ton/hr]" osv.
    period='2050-2055',          # None => siste periode (f.eks. 2050-2055)
    agg="max",            # viktig for “bare én retning”
    color="tab:blue",
)

