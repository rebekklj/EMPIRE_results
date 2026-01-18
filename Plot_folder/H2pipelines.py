from pathlib import Path
from typing import Optional, Dict, Tuple, Union, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# -----------------------------
# Felles: finn + last result-csv
# -----------------------------

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
    """
    Lager eksport→import matrise (y=FromNode, x=ToNode) for en generisk "Value".

    mode:
      - "mean"           : gjennomsnitt av Value (per representant-time)
      - "sum"            : sum av Value over rader (mest for feilsjekk)
      - "expected_total" : forventet total over året:
            Value * season_scale[Season] * p(Scenario)*p(GasScenario)
            -> for MW blir dette MWh
            -> for H2_ton_per_h blir dette ton
      - "expected_TWh"   : som expected_total, men konvertert til TWh:
            MW: MWh / 1e6
            H2 ton: ton * h2_mwh_per_ton / 1e6
    """
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
