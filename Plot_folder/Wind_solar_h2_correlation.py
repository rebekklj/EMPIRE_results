# pem_solar_wind_corr_pycharm.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
RESULTS_DIR= data_dir / 'Results_FINAL_BASE_FLEX_emcap_cyclelim' / 'full_model_base'

OUTDIR = None  # eller: Path(r"C:\path\til\output")

# Valgfritt: filtrer
FILTER_PERIOD = '2050-2055'   # f.eks. "2030" eller 2030
FILTER_NODE = ('Portugal')     # f.eks. "NO1"

# Velg hvilken PEM-serie som brukes
PEM_MODE = "green"     # "total" | "green" | "yellow" | "import"

# Hvis du vil velge mappe via fil-dialog i stedet for å hardkode:
USE_FOLDER_DIALOG = False
# =========================


KEYS = ["Node", "Period", "Scenario", "GasScenario", "Season", "Hour"]


def _find_cols(df: pd.DataFrame, needle: str) -> list[str]:
    needle = needle.lower()
    return [c for c in df.columns if c.lower().endswith("_mw") and needle in c.lower()]


def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def heatmap(corr: pd.DataFrame, title: str, outpath: Path):
    plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.colorbar(label="Correlation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def scatter_with_fit(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask].astype(float), y[mask].astype(float)

    plt.figure()
    plt.scatter(x2, y2, s=8)

    if len(x2) >= 2:
        m, b = np.polyfit(x2, y2, 1)
        xs = np.linspace(x2.min(), x2.max(), 100)
        plt.plot(xs, m * xs + b)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def maybe_choose_folder(initial: Path | None = None) -> Path:
    # Bruker tkinter for å velge folder hvis ønskelig
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(initialdir=str(initial) if initial else None, title="Velg resultatmappe")
    if not folder:
        raise RuntimeError("Ingen mappe valgt.")
    return Path(folder)


def main():
    global RESULTS_DIR

    if USE_FOLDER_DIALOG:
        RESULTS_DIR = maybe_choose_folder(RESULTS_DIR if RESULTS_DIR else None)

    if RESULTS_DIR is None:
        raise ValueError("RESULTS_DIR er ikke satt.")

    results_dir = Path(RESULTS_DIR)
    elec_path = results_dir / "results_elec_generation_operational.csv"
    h2_path = results_dir / "results_hydrogen_production.csv"

    if not elec_path.exists():
        raise FileNotFoundError(f"Mangler {elec_path}")
    if not h2_path.exists():
        raise FileNotFoundError(f"Mangler {h2_path}")

    outdir = Path(OUTDIR) if OUTDIR else (results_dir / "plots_pem_corr")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Les filer ---
    elec = pd.read_csv(elec_path,index_col=False)
    h2 = pd.read_csv(h2_path,index_col=False)

    print("\n=== ELEC HEAD ===")
    print(elec[KEYS].head(3))
    print("\n=== ELEC DTYPES (KEYS) ===")
    print(elec[KEYS].dtypes)

    print("\n=== H2 HEAD ===")
    print(h2[KEYS].head(3))
    print("\n=== H2 DTYPES (KEYS) ===")
    print(h2[KEYS].dtypes)

    # --- Finn sol og vind kolonner automatisk ---
    solar_cols = _find_cols(elec, "solar")
    wind_cols = _find_cols(elec, "wind")

    if not solar_cols:
        raise ValueError("Fant ingen sol-kolonner (*solar* + _MW) i results_elec_generation_operational.csv.")
    if not wind_cols:
        raise ValueError("Fant ingen vind-kolonner (*wind* + _MW) i results_elec_generation_operational.csv.")

    elec = _safe_to_numeric(elec, solar_cols + wind_cols)
    elec["solar_MW"] = elec[solar_cols].sum(axis=1)
    elec["wind_MW"] = elec[wind_cols].sum(axis=1)
    elec_small = elec[KEYS + ["solar_MW", "wind_MW"]].copy()

    # --- PEM hydrogen produksjon ---
    pem_y = "PEM_yellow production [ton]"
    pem_g = "PEM_green production [ton]"
    pem_i = "PEM_import production [ton]"

    for c in [pem_y, pem_g, pem_i]:
        if c not in h2.columns:
            raise ValueError(f"Mangler kolonne i hydrogenfila: {c}")

    h2 = _safe_to_numeric(h2, [pem_y, pem_g, pem_i])

    if PEM_MODE == "yellow":
        h2["pem_ton"] = h2[pem_y]
    elif PEM_MODE == "green":
        h2["pem_ton"] = h2[pem_g]
    elif PEM_MODE == "import":
        h2["pem_ton"] = h2[pem_i]
    else:
        h2["pem_ton"] = h2[pem_y].fillna(0) + h2[pem_g].fillna(0) + h2[pem_i].fillna(0)

    h2_small = h2[KEYS + ["pem_ton"]].copy()

    # --- Merge ---
    df = elec_small.merge(h2_small, on=KEYS, how="inner")

    print("\nSjekk variasjon:")
    print(df[["solar_MW", "wind_MW", "pem_ton"]].describe())
    print("\nAntall unike verdier:")
    print(df[["solar_MW", "wind_MW", "pem_ton"]].nunique())
    print("\nAndel null:")
    print((df[["solar_MW", "wind_MW", "pem_ton"]] == 0).mean())

    # --- Filtrering ---
    if FILTER_PERIOD is not None:
        df = df[df["Period"].astype(str) == str(FILTER_PERIOD)]
    if FILTER_NODE is not None:
        df = df[df["Node"].astype(str) == str(FILTER_NODE)]

    if df.empty:
        raise ValueError("Ingen rader etter merge/filtrering. Sjekk at Node/Period finnes i begge filer.")

    cols = ["solar_MW", "wind_MW", "pem_ton"]

    pear = df[cols].corr(method="pearson")
    spear = df[cols].corr(method="spearman")

    pear.to_csv(outdir / "corr_overall_pearson.csv")
    spear.to_csv(outdir / "corr_overall_spearman.csv")

    # per node
    rows = []
    for node, g in df.groupby("Node"):
        if len(g) < 3:
            continue
        rows.append({
            "Node": node,
            "pear_solar_pem": g["solar_MW"].corr(g["pem_ton"], method="pearson"),
            "pear_wind_pem": g["wind_MW"].corr(g["pem_ton"], method="pearson"),
            "pear_solar_wind": g["solar_MW"].corr(g["wind_MW"], method="pearson"),
            "spear_solar_pem": g["solar_MW"].corr(g["pem_ton"], method="spearman"),
            "spear_wind_pem": g["wind_MW"].corr(g["pem_ton"], method="spearman"),
            "spear_solar_wind": g["solar_MW"].corr(g["wind_MW"], method="spearman"),
            "n": len(g),
        })
    per_node = pd.DataFrame(rows).sort_values(["n", "Node"], ascending=[False, True])
    per_node.to_csv(outdir / "corr_per_node.csv", index=False)

    # Plot
    heatmap(pear, "Overall Pearson correlation", outdir / "heatmap_overall_pearson.png")
    heatmap(spear, "Overall Spearman correlation", outdir / "heatmap_overall_spearman.png")

    scatter_with_fit(df["solar_MW"], df["pem_ton"],
                     "PEM production vs Solar production",
                     "Solar production [MW]", "PEM production [ton]",
                     outdir / "scatter_pem_vs_solar.png")

    scatter_with_fit(df["wind_MW"], df["pem_ton"],
                     "PEM production vs Wind production",
                     "Wind production [MW]", "PEM production [ton]",
                     outdir / "scatter_pem_vs_wind.png")

    scatter_with_fit(df["solar_MW"], df["wind_MW"],
                     "Wind vs Solar production",
                     "Solar production [MW]", "Wind production [MW]",
                     outdir / "scatter_wind_vs_solar.png")

    print(f"Ferdig. Output: {outdir}")
    print(f"Fant {len(solar_cols)} sol-kolonner og {len(wind_cols)} vind-kolonner.")
    print("Eksempel sol-kolonner:", solar_cols[:5])
    print("Eksempel vind-kolonner:", wind_cols[:5])


if __name__ == "__main__":
    main()



