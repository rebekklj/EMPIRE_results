# Plot_folder/plot_results.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from functions_plot import Expected_annual_production, plot_top_map, P_prodVSimport_piechart, Plot_Installed_capacity_per_tech_split,HydrogenProd_piechart,plot_power_demand,plot_hydrogen_use


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_w_el_demand" / "full_model_base"
plot_dir = data_dir / "Results_w_el_demand" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer =False
figurnavn = "wElDemand"

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

P_prodVSimport_piechart(Power_balance, 12, 2, savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

plot_top_map(Elec_generation_inv,3,savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

# In[]
hydrogen_production=pd.read_csv(result_dir/ 'results_hydrogen_production.csv')
HydrogenProd_piechart(hydrogen_production, 12, 2, savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

# In[]
Power_balance['Power reformer plant [MWh]']=Power_balance['Power reformer plant [MWh]']*(-1)
plot_power_demand(Power_balance,n_hours=12)

# In[]
hydrogen_use=pd.read_csv(result_dir/ 'results_hydrogen_use.csv')

plot_hydrogen_use(hydrogen_use, 12, 2, savefigure=False, figurename=None, results_dir=None)

















