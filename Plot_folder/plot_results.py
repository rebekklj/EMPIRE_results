#Plot_folder/plot_results.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from src.functions_plot import Expected_annual_production, plot_top_map, P_prodVSimport_piechart, Plot_Installed_capacity_per_tech_split,HydrogenProd_piechart,plot_power_demand,plot_hydrogen_use, make_pipeline_summary, Yearly_hydrogenProd_perTech, plot_h2_demand_stacked_area, H2prod_per_country, plot_h2_storage


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
#result_dir = data_dir / "Results_PEM_types2_0611" /"Results_PEM_types2"/"full_model_base"
#plot_dir = data_dir / "Results_PEM_types2_0611" #lagrer figurene i resultat mappen
result_dir = data_dir/"Results_BASE"/"full_model_base"
plot_dir = data_dir/"Results_BASE"
plot_dir.mkdir(exist_ok=True)

Lagre_figurer =True
figurnavn = "base"

# In[]
'''
# -----------------  Expected annual production ------------------
Elec_generation_inv = pd.read_csv(result_dir / "results_elec_generation_inv.csv")

Expected_annual_production(
    Elec_generation_inv,
    savefigure=Lagre_figurer,
    results_dir=plot_dir,
    figurename=figurnavn
)

Europe_summary = pd.read_csv((result_dir/ "results_output_EuropeSummary.csv"), delimiter=",",skiprows=16, usecols=[0, 1, 2, 3, 4, 5], skipfooter=16, engine='python')
Europe_summary["genExistingCap_MW"] = (Europe_summary["genInstalledCap_MW"] - Europe_summary["genInvCap_MW"])


#------------------- Installed capacity per tech -----------------------
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

#--------------------- Hydrogen production pie chart --------------
hydrogen_production=pd.read_csv(result_dir/ 'results_hydrogen_production.csv')
HydrogenProd_piechart(hydrogen_production, 12, 2, savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

# -------------------- Power demand, stacked bar plot ---------------
# In[]
Power_balance= pd.read_csv(result_dir / "results_power_balance.csv")
Power_balance['Power reformer plant [MWh]']=Power_balance['Power reformer plant [MWh]']*(-1) # kva er hensikten med denne ?
plot_power_demand(Power_balance,n_hours=12, savefigure=True, figurename=figurnavn, results_dir=plot_dir)


# ------------------- Hydrogen demand pr sector, area graph ------------
# In[]
hydrogen_use=pd.read_csv(result_dir/ 'results_hydrogen_use.csv')

plot_hydrogen_use(hydrogen_use, 12, 2, savefigure=True, figurename=figurnavn, results_dir=plot_dir)
'''
# ---------------- Hydrogen pipeline csv and plot of top 5 importers and exporters pr period -----------------
# Pipeline csv and plots
palette = ['orchid','teal', 'darkseagreen',
            'khaki', 'plum', 'darkslateblue',
            'lavender', 'lightskyblue','mediumslateblue', 'violet']

pipelines = make_pipeline_summary(result_dir/"results_hydrogen_pipeline_operational.csv" ,
                              n_hours=12,
                              output_dir=plot_dir/"summary_pipeline.csv",
                              export_color_map = palette,
                              import_color_map = palette,
                              color_cycle = palette)

hydrogen_production = pd.read_csv(result_dir/'results_hydrogen_production.csv')
'''
#
h2_prod = Yearly_hydrogenProd_perTech(
    hydrogen_production,
    x1="PEM_blue production [ton]",
    x2="PEM_green production [ton]",
    x3="PEM_grey production [ton]",
    x4="ALK production [ton]",
    x5 ="SOEC production [ton]",
    x6 ="Reformer production [ton]",
    n_scen=2,
    n_hours=12,
    figurename=figurnavn,
    results_dir= plot_dir,
    savefigure=True
)

# ------------------ Total hydrogen production per country, horizontal barplot -----------
hydrogen_production = pd.read_csv(result_dir/'results_hydrogen_production.csv')
h2_per_country = H2prod_per_country(hydrogen_production,
                   n_hours=12,
                   n_scen=2,
                   figurename=figurnavn,
                   results_dir=plot_dir,
                   savefigure=True
                   )
'''


#plot_h2_storage(result_dir/'results_hydrogen_storage_operational.csv', n_hours=12, n_scen=2)






