# Plot_folder/plot_results.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from functions_plot import Expected_annual_production, plot_top_map, P_prodVSimport_piechart, Plot_Installed_capacity_per_tech_split,HydrogenProd_piechart


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_noTransmissionBuilt" / "full_model_base"
plot_dir = data_dir / "Results_noTransmissionBuilt" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer = True
figurnavn = "noTransmissionBuilt"

# In[]

Elec_generation_inv = pd.read_csv(result_dir / "results_elec_generation_inv.csv")

Expected_annual_production(
    Elec_generation_inv,
    savefigure=Lagre_figurer,
    results_dir=plot_dir,
    figurename=figurnavn
)

Europe_summary = pd.read_csv((result_dir/ "results_output_EuropeSummary.csv"), delimiter=",",skiprows=16, usecols=[0, 1, 2, 3, 4, 5], skipfooter=16, engine='python')
Europe_summary["genExistingCap_MW"] = (Europe_summary["genInstalledCap_MW"] - Europe_summary["genInvCap_MW"])

Plot_Installed_capacity_per_tech_split(Europe_summary,
                                           threshold=90_000,
                                           figsize=(18, 12),
                                           savefigure=Lagre_figurer,
                                           figurename=figurnavn, results_dir=plot_dir)

# In[]
Power_balance= pd.read_csv(result_dir / "results_power_balance.csv")

P_prodVSimport_piechart(Power_balance, 12, 2, savefigure=False, figurename=None, results_dir=None)

plot_top_map(Elec_generation_inv,3)

# In[]
hydrogen_production=pd.read_csv(result_dir/ 'results_hydrogen_production.csv')
HydrogenProd_piechart(hydrogen_production, 12, 2, savefigure=False, figurename=None, results_dir=None)







