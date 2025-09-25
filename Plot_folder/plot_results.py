# Plot_folder/plot_results.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from functions_plot import Expected_annual_production, plot_top_map

project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_noTransmissionBuilt" / "full_model_base"
plot_dir = data_dir / "Results_noTransmissionBuilt" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer = True
figurnavn = "noTransmissionBuilt"

# Les resultater
Elec_generation_inv = pd.read_csv(result_dir / "results_elec_generation_inv.csv")

# Kall på plottfunksjonen
Expected_annual_production(
    Elec_generation_inv,
    savefigure=Lagre_figurer,
    results_dir=plot_dir,
    figurename=figurnavn
)





