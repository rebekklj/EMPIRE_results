# Plot_folder/plot_results.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.functions_plot import (Expected_annual_production, plot_top_map, P_prodVSimport_piechart, Plot_Installed_capacity_per_tech_split,HydrogenProd_piechart, Yearly_hydrogenProd_perTech, plot_insCap_top_nodes)


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_trans_inv_2409" / "full_model_base"
plot_dir = data_dir / "Results_trans_inv_2409" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer = True
figurnavn = "withTransmissionBuilt"



Elec_generation_inv = pd.read_csv(result_dir / "results_elec_generation_inv.csv")


# Plot av annual production
'''
Expected_annual_production(
    Elec_generation_inv,
    savefigure=Lagre_figurer,
    results_dir=plot_dir,
    figurename=figurnavn
)
'''
Europe_summary = pd.read_csv((result_dir/ "results_output_EuropeSummary.csv"), delimiter=",",skiprows=16, usecols=[0, 1, 2, 3, 4, 5], skipfooter=16, engine='python')
Europe_summary["genExistingCap_MW"] = (Europe_summary["genInstalledCap_MW"] - Europe_summary["genInvCap_MW"])


# PLot av installed capacity
'''

Plot_Installed_capacity_per_tech_split(Europe_summary,
                                           threshold=90_000,
                                           figsize=(18, 12),
                                           savefigure=Lagre_figurer,
                                           figurename=figurnavn, results_dir=plot_dir)
'''

# Plot of import vs export map

Power_balance= pd.read_csv(result_dir / "results_power_balance.csv")

'''
P_prodVSimport_piechart(Power_balance, 12, 2, savefigure=Lagre_figurer,
                                           figurename=figurnavn, results_dir=plot_dir)

#plot_top_map(Elec_generation_inv,3, savefigure=Lagre_figurer,
#                                           figurename=figurnavn, results_dir=plot_dir)
'''

# Barchart of hydrogen production technologies
# In[]
hydrogen_production=pd.read_csv(result_dir/ 'results_hydrogen_production.csv')
#HydrogenProd_piechart(hydrogen_production, 12, 2, savefigure=Lagre_figurer,
#                                           figurename=figurnavn, results_dir=plot_dir)

'''
Yearly_hydrogenProd_perTech(hydrogen_production, n_hours=12, n_scen = 2 , x1='PEM production [ton]', x2="ALK production [ton]", x3='SOEC production [ton]', x4='Reformer production [ton]')
'''

### Line graph of installed capacity of chosen technology/generator for top n nodes


palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
           "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2df8a"]
el_gen_inv = pd.read_csv(result_dir/'results_elec_generation_inv.csv')

plot_insCap_top_nodes(el_gen_inv,tech="Solar", n_top=5, unit="GW", savepath=plot_dir /"pv_inscap.png", colors=palette)
#plt.show()

#

