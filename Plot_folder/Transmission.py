
# -----------------------------
# Paths
# -----------------------------
project_dir = Path(__file__).resolve().parents[1]  # EMPIRE_results_git
data_dir = project_dir / "data"

result_dir1 = data_dir / "Results_FINAL_BASE_NOFLEX_emcap_cyclelim" / "full_model_base"
result_dir2 = data_dir / "Results_FINAL_optimistic_price_scen" / "full_model_base"

# -----------------------------
# 1) Lesing av resultater
# -----------------------------

def find_transmission_operational_files(
    base_dir: Union[str, Path],
    filename: str = "results_elec_transmission_operational.csv",
) -> list[Path]:
    """
    Finn alle results_elec_transmission_operational.csv rekursivt under base_dir.
    Typisk: base_dir = mappen der du har mange "run"-mapper.
    """
    base = Path(base_dir).expanduser().resolve()
    return sorted(base.rglob(filename))


def load_transmission_operational(
    base_dir: Union[str, Path],
    filename: str = "results_elec_transmission_operational.csv",
) -> pd.DataFrame:
    """
    Leser alle results_elec_transmission_operational.csv under base_dir og setter:
      - __run__  = foreldremappenavn til csv (ofte run-id)
      - __path__ = full filsti
    """
    files = find_transmission_operational_files(base_dir, filename=filename)
    if not files:
        raise FileNotFoundError(f"Fant ingen '{filename}' under {Path(base_dir).resolve()}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["__run__"] = f.parent.name
        df["__path__"] = str(f)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Sikre numerikk der det trengs
    out["TransmissionReceived_MW"] = pd.to_numeric(out["TransmissionReceived_MW"], errors="coerce")
    out["Losses_MW"] = pd.to_numeric(out["Losses_MW"], errors="coerce")
    out = out.dropna(subset=["FromNode", "ToNode", "TransmissionReceived_MW"])

    # Period kan være int eller str i csv; gjør til str for robust filtrering
    out["Period"] = out["Period"].astype(str).str.strip()

    return out


# -----------------------------
# 2) Aggregasjon til eksport→import matrise
# -----------------------------

def _normalize_weights_from_unique(values: Iterable, provided: Optional[Dict] = None) -> Dict:
    """
    Hvis provided ikke er gitt: antas lik sannsynlighet for alle unike verdier.
    """
    uniq = sorted(set(values))
    if not uniq:
        return {}

    if provided is None:
        p = 1.0 / len(uniq)
        return {u: p for u in uniq}

    # Hvis provided er gitt men mangler noen keys, fyll med 0
    w = {u: float(provided.get(u, 0.0)) for u in uniq}

    # Normaliser hvis summen ikke er 1 (typisk greit å gjøre)
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w


def transmission_activity_matrix(
    df: pd.DataFrame,
    period: Union[int, str],
    run: Optional[str] = None,
    mode: str = "expected_TWh",
    *,
    include_losses: bool = False,
    season_scale: Optional[Dict[str, float]] = None,
    scenario_prob: Optional[Dict] = None,
    gas_scenario_prob: Optional[Dict] = None,
    nodes: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Lager en matrise (y=eksport/FromNode, x=import/ToNode).

    mode:
      - "mean_MW"      : gjennomsnittlig MW per representant-time (over alle rader)
      - "sum_MW"       : sum MW over rader (mest nyttig for feilsjekk)
      - "expected_TWh" : forventet energi i TWh, dvs:
            MW * season_scale[Season] -> MWh
            vektet med scenario_prob og gas_scenario_prob
            summeres og konverteres til TWh (MWh / 1e6)

    season_scale:
      dict som map'er Season -> antall timer sesongen representerer (seasScale i modellen).
      Hvis ikke gitt: 1.0 for alle seasons (da blir "expected_TWh" mer som "vektet sum av representant-timer").

    scenario_prob / gas_scenario_prob:
      dict som map'er Scenario/GasScenario -> sannsynlighet.
      Hvis ikke gitt: antas lik sannsynlighet for alle unike verdier i data.

    include_losses:
      Hvis True bruker (TransmissionReceived_MW + Losses_MW) som "aktivitet".
      Ellers bruker bare TransmissionReceived_MW (det er det som skrives som mottatt effekt). :contentReference[oaicite:1]{index=1}
    """
    if df.empty:
        raise ValueError("Input-dataframe er tom.")

    d = df.copy()

    # Filtrer run hvis ønsket
    if run is not None:
        d = d[d["__run__"].astype(str) == str(run)]
        if d.empty:
            raise ValueError(f"Ingen rader etter run-filter: run='{run}'")

    # Filtrer period
    d = d[d["Period"] == str(period)]
    if d.empty:
        raise ValueError(f"Ingen rader for Period='{period}'")

    # Velg "aktivitet"
    if include_losses:
        if "Losses_MW" not in d.columns:
            raise ValueError("Fant ikke Losses_MW-kolonnen, men include_losses=True.")
        d["__mw__"] = d["TransmissionReceived_MW"] + d["Losses_MW"]
    else:
        d["__mw__"] = d["TransmissionReceived_MW"]

    if mode == "mean_MW":
        agg = d.groupby(["FromNode", "ToNode"], as_index=False)["__mw__"].mean()
        agg = agg.rename(columns={"__mw__": "value"})
        unit = "MW (gjennomsnitt)"
    elif mode == "sum_MW":
        agg = d.groupby(["FromNode", "ToNode"], as_index=False)["__mw__"].sum()
        agg = agg.rename(columns={"__mw__": "value"})
        unit = "MW (sum over rader)"
    elif mode == "expected_TWh":
        # season_scale
        season_scale = season_scale or {}
        if "Season" in d.columns:
            d["__seas__"] = d["Season"].astype(str)
            d["__seas_scale__"] = d["__seas__"].map(season_scale).fillna(1.0)
        else:
            d["__seas_scale__"] = 1.0

        # scenario/gas weights (likt hvis ikke oppgitt)
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

        # MW * (timer) = MWh, vektet, sum, til TWh
        d["__wmwh__"] = d["__mw__"] * d["__seas_scale__"] * d["__p_s__"] * d["__p_g__"]
        agg = d.groupby(["FromNode", "ToNode"], as_index=False)["__wmwh__"].sum()
        agg["value"] = agg["__wmwh__"] / 1e6  # MWh -> TWh
        agg = agg.drop(columns=["__wmwh__"])
        unit = "TWh (forventet)"
    else:
        raise ValueError("mode må være en av: 'mean_MW', 'sum_MW', 'expected_TWh'")

    # node-rekkefølge
    if nodes is None:
        nodes = sorted(set(agg["FromNode"].astype(str)).union(set(agg["ToNode"].astype(str))))
    else:
        nodes = [str(n) for n in nodes]

    mat = (
        agg.assign(FromNode=agg["FromNode"].astype(str), ToNode=agg["ToNode"].astype(str))
           .pivot_table(index="FromNode", columns="ToNode", values="value", aggfunc="sum", fill_value=0.0)
           .reindex(index=nodes, columns=nodes, fill_value=0.0)
    )

    return mat, unit


def drop_all_zero_rows_cols(mat: pd.DataFrame, tol: float = 0.0) -> pd.DataFrame:
    """
    Fjerner rader og kolonner som bare består av 0.
    tol > 0 gjør at vi behandler |verdi| <= tol som 0 (nyttig ved små numeriske rester).
    """
    a = mat.to_numpy()

    if tol > 0:
        nonzero_row = (np.abs(a) > tol).any(axis=1)
        nonzero_col = (np.abs(a) > tol).any(axis=0)
    else:
        nonzero_row = (a != 0).any(axis=1)
        nonzero_col = (a != 0).any(axis=0)

    return mat.loc[nonzero_row, nonzero_col]

# -----------------------------
# 3) Plotting
# -----------------------------

def plot_transmission_heatmap(
    matrix: pd.DataFrame,
    title: str = "Power Export Flows",
    *,
    cmap: str = "Blues",
    annotate: bool = True,
    annotate_fmt: str = ".2f",
    figsize_scale: float = 1.0,
    savepath: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plotter heatmap: y=eksport (index), x=import (columns).
    Returnerer (fig, ax) så du kan videre-tilpasse i ditt script.
    """
    n = matrix.shape[0]

    # Skaler figur så labels får plass
    w = max(10.0, n * 0.35) * figsize_scale
    h = max(8.0, n * 0.28) * figsize_scale

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("Import region")
    ax.set_ylabel("Export region")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    if annotate and n <= 60:
        vals = matrix.values
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(vals[i, j], annotate_fmt), ha="center", va="center", fontsize=6)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath).expanduser().resolve()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200)

    return fig, ax


def plot_transmission_activity_from_folder(
    base_dir: Union[str, Path],
    period: Union[int, str],
    *,
    run: Optional[str] = None,
    mode: str = "expected_TWh",
    include_losses: bool = False,
    season_scale: Optional[Dict[str, float]] = None,
    scenario_prob: Optional[Dict] = None,
    gas_scenario_prob: Optional[Dict] = None,
    annotate: bool = True,
    annotate_fmt: str = ".2f",
    figsize_scale: float = 1.0,
    savepath: Optional[Union[str, Path]] = None,
    title_prefix: str = "Power Export Flows",
    drop_zeros: bool = True,zero_tol: float = 0.0,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    "En-linjes" funksjon: les -> bygg matrise -> plott.
    Returnerer (matrix, fig, ax).
    """
    df = load_transmission_operational(base_dir)
    mat, unit = transmission_activity_matrix(
        df,
        period=period,
        run=run,
        mode=mode,
        include_losses=include_losses,
        season_scale=season_scale,
        scenario_prob=scenario_prob,
        gas_scenario_prob=gas_scenario_prob,
    )

    if drop_zeros:
        mat = drop_all_zero_rows_cols(mat, tol=zero_tol)

    title = f"{title_prefix} ({unit}) i {period}"
    fig, ax = plot_transmission_heatmap(
        mat,
        title=title,
        annotate=annotate,
        annotate_fmt=annotate_fmt,
        figsize_scale=figsize_scale,
        savepath=savepath,
    )
    return mat, fig, ax

seasScale = {
    "winter": 26.0,
    "spring": 26.0,
    "summer": 26.0,
    "fall": 26.0,
    "peak1": 12,
    "peak2": 12,
}



mat, fig, ax = plot_transmission_activity_from_folder(
    base_dir=result_dir1,
    period='2050-2055',
    mode="expected_TWh",
    season_scale=seasScale,
    annotate=True,
)
plt.show()



from matplotlib.colors import TwoSlopeNorm


# --- (forutsetter at du allerede har disse fra tidligere) ---
# - load_transmission_operational(base_dir) -> pd.DataFrame
# - transmission_activity_matrix(df, period, ...) -> (mat, unit)
# - drop_all_zero_rows_cols(mat, tol=...)
# -----------------------------------------------------------


def align_square_matrices(m1: pd.DataFrame, m2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sørger for at begge matriser har samme (union) index/columns i samme rekkefølge.
    Antar at de er "square" (noder x noder).
    """
    nodes = sorted(set(m1.index).union(m1.columns).union(m2.index).union(m2.columns))
    m1a = m1.reindex(index=nodes, columns=nodes, fill_value=0.0)
    m2a = m2.reindex(index=nodes, columns=nodes, fill_value=0.0)
    return m1a, m2a


def select_top_nodes_by_absdiff(mat_diff: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
    """
    For mange noder blir uleselig. Denne velger topp_k noder med størst "aktivitet i differanse",
    basert på sum av absolutte forskjeller i rad + kolonne.
    """
    if top_k is None:
        return mat_diff

    n = mat_diff.shape[0]
    if top_k <= 0 or top_k >= n:
        return mat_diff

    a = np.abs(mat_diff.to_numpy())
    score = a.sum(axis=1) + a.sum(axis=0)
    idx = np.argsort(score)[::-1][:top_k]
    nodes = mat_diff.index.to_numpy()[idx]
    nodes = list(map(str, nodes))
    return mat_diff.loc[nodes, nodes]


def transmission_difference_matrix_from_folders(
    result_dir1: Union[str, Path],
    result_dir2: Union[str, Path],
    period: Union[int, str],
    *,
    run: Optional[str] = None,
    mode: str = "expected_TWh",
    include_losses: bool = False,
    season_scale: Optional[Dict[str, float]] = None,
    scenario_prob: Optional[Dict] = None,
    gas_scenario_prob: Optional[Dict] = None,
    drop_zeros: bool = True,
    zero_tol: float = 0.0,
    top_k: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Returnerer differanse-matrise = mat2 - mat1 (dir2 minus dir1).
    Positiv verdi betyr mer flyt i result_dir2 enn i result_dir1.
    """
    df1 = load_transmission_operational(result_dir1)
    df2 = load_transmission_operational(result_dir2)

    mat1, unit = transmission_activity_matrix(
        df1, period=period, run=run, mode=mode,
        include_losses=include_losses,
        season_scale=season_scale,
        scenario_prob=scenario_prob,
        gas_scenario_prob=gas_scenario_prob,
    )
    mat2, _ = transmission_activity_matrix(
        df2, period=period, run=run, mode=mode,
        include_losses=include_losses,
        season_scale=season_scale,
        scenario_prob=scenario_prob,
        gas_scenario_prob=gas_scenario_prob,
    )

    mat1, mat2 = align_square_matrices(mat1, mat2)
    diff = mat2 - mat1

    if drop_zeros:
        diff = drop_all_zero_rows_cols(diff, tol=zero_tol)

    # valgfritt: begrens til "viktigste" noder i differansen for lesbarhet
    diff = select_top_nodes_by_absdiff(diff, top_k=top_k)

    return diff, unit


def plot_transmission_difference_heatmap(
    mat_diff: pd.DataFrame,
    *,
    title: str,
    unit: str,
    cmap: str = "RdBu_r",
    figsize_scale: float = 1.0,
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show_tick_every: Optional[int] = None,
    label_fontsize: Optional[int] = None,
    grid: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plotter differanse-heatmap med divergerende colormap sentrert på 0.
    For mange noder: vi "tynner" tick labels automatisk (eller via show_tick_every).
    """
    n = mat_diff.shape[0]
    vals = mat_diff.to_numpy()

    # Symmetrisk fargeskala rundt 0
    vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Figur-størrelse: skaler opp mer aggressivt enn før (for lesbarhet)
    w = max(12.0, n * 0.40) * figsize_scale
    h = max(10.0, n * 0.33) * figsize_scale

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel("Import node",fontsize=12)
    ax.set_ylabel("Export node",fontsize=12)

    # Auto-thinning av tick labels (unngå at alt blir en grå klump)
    if show_tick_every is None:
        # prøv å holde ~max 45 labels på hver akse
        show_tick_every = max(1, n // 45)

    xticks = np.arange(0, n, show_tick_every)
    yticks = np.arange(0, n, show_tick_every)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    cols = mat_diff.columns.to_list()
    rows = mat_diff.index.to_list()

    if label_fontsize is None:
        # litt dynamisk font
        label_fontsize = 12 if n <= 60 else 7

    ax.set_xticklabels([cols[i] for i in xticks], rotation=90, ha="center", fontsize=label_fontsize)
    ax.set_yticklabels([rows[i] for i in yticks], fontsize=label_fontsize)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Difference (TWh)  [BCONSTANT - BFLEX]", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=label_fontsize)

    if grid and n <= 80:
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath).expanduser().resolve()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_transmission_difference_from_folders(
    result_dir1: Union[str, Path],
    result_dir2: Union[str, Path],
    period: Union[int, str],
    *,
    run: Optional[str] = None,
    mode: str = "expected_TWh",
    include_losses: bool = False,
    season_scale: Optional[Dict[str, float]] = None,
    scenario_prob: Optional[Dict] = None,
    gas_scenario_prob: Optional[Dict] = None,
    drop_zeros: bool = True,
    zero_tol: float = 0.0,
    top_k: Optional[int] = 80,
    figsize_scale: float = 1.2,
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show_tick_every: Optional[int] = None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    One-liner: lager differanse-matrise og plotter den.
    """
    diff, unit = transmission_difference_matrix_from_folders(
        result_dir1, result_dir2, period,
        run=run, mode=mode, include_losses=include_losses,
        season_scale=season_scale,
        scenario_prob=scenario_prob, gas_scenario_prob=gas_scenario_prob,
        drop_zeros=drop_zeros, zero_tol=zero_tol, top_k=top_k,
    )

    title = f"Transmission difference (dir2 - dir1) in {period}"
    fig, ax = plot_transmission_difference_heatmap(
        diff,
        title=title,
        unit=unit,
        figsize_scale=figsize_scale,
        savepath=savepath,
        dpi=dpi,
        show_tick_every=show_tick_every,
    )
    return diff, fig, ax

season_scale = {
    "winter": 26.0, "spring": 26.0, "summer": 26.0, "fall": 26.0,
    "peak1": 12.0, "peak2": 12.0,
}

diff, fig, ax = plot_transmission_difference_from_folders(
    result_dir1, result_dir2,
    period='2050-2055',
    mode="expected_TWh",
    season_scale=season_scale,
    drop_zeros=True,
    zero_tol=1e-9,
    top_k=25,                # sett None for ALLE noder (men det blir ofte uleselig)
    figsize_scale=1.4,       # større figur
    dpi=350,                 # skarpere fil
    savepath="out/trans_diff_2050.png",
)

plt.show()