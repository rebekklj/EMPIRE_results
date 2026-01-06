from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from functions_plot import (Expected_annual_production, plot_top_map,
                            P_prodVSimport_piechart, Plot_Installed_capacity_per_tech_split,
                            HydrogenProd_piechart, plot_power_demand, plot_hydrogen_use,
                            H2prod_per_country, plot_hydrogen_capacity, plot_OBJ_generator_inv,
                            plot_H2_costs_per_period, plot_DGF_charge_discharge_stochastic,
                            HydrogenStorage_scatter, plot_DGF_charge_discharge_stochastic,
                            plot_discharge_cycles_sawtooth,plot_power_balance_for_high_h2,
                            hydrogen_prod_vs_import_bar,Yearly_hydrogenProd_perTech,
                            plot_storage_charge_discharge_total,plot_h2_prod_discharge_plus_demand)


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_FINAL_BASE_FLEX_emcap" / "full_model_base"
plot_dir = data_dir / "Results_FINAL_BASE_FLEX_emcap" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer =False
figurnavn = "BASE_moderate"

gen_info=('yes')
H2_prod='yes'
el_demand='no'
H2_storage='yes'

# In[]
if gen_info=='yes':
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
if gen_info=='yes':
    P_prodVSimport_piechart(Power_balance, 12, 2, savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

    plot_top_map(Elec_generation_inv,3,savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

# In[]
hydrogen_production=pd.read_csv(result_dir/ 'results_hydrogen_production.csv')
if H2_prod=='yes':
    HydrogenProd_piechart(hydrogen_production, 12, 2, savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)
    Yearly_hydrogenProd_perTech(hydrogen_production, 'PEM_green production [ton]', 'PEM_yellow production [ton]','PEM_import production [ton]',
                                'ALK production [ton]', 'SOEC production [ton]', 'Reformer production [ton]', 2, 12, 55)

# In[]

if el_demand=='yes':
    Power_balance['Power reformer plant [MWh]'] = Power_balance['Power reformer plant [MWh]'] * (-1)
    plot_power_demand(Power_balance,n_hours=12)

# In[]
hydrogen_use=pd.read_csv(result_dir/ 'results_hydrogen_use.csv')
if H2_prod=='yes':
    plot_hydrogen_use(hydrogen_use, 12, 2, savefigure=False, figurename=None, results_dir=None)
    hydrogen_prod_vs_import_bar(hydrogen_use)




# In[]
if H2_storage=='yes':

    Gen_inv=pd.read_csv(result_dir/ 'results_objective_components_generation_inv_costs.csv')
    Gen_op=pd.read_csv(result_dir/ 'results_objective_components_operational_costs.csv')
    H2_inv=pd.read_csv(result_dir/ 'results_hydrogen_costs.csv')
    Stor_el=pd.read_csv(result_dir/ 'results_objective_components_storage_inv_costs.csv')
    Trans_inv=pd.read_csv(result_dir/ 'results_objective_components_transmission_inv_costs.csv')
    plot_OBJ_generator_inv(Gen_op,Gen_inv,H2_inv,Stor_el,Trans_inv,2)

    plot_H2_costs_per_period(H2_inv)

    H2_storage=pd.read_csv(result_dir/ 'results_hydrogen_storage_inv.csv')

    periods=["2020-2025", "2025-2030", "2030-2035", "2035-2040",
                   "2040-2045", "2045-2050", "2050-2055"]

    for period in periods:
        HydrogenStorage_scatter(H2_storage, period, savefigure=False, figurename=None, results_dir=None)

    df = pd.read_csv(result_dir/"results_hydrogen_storage_operational.csv")
    plot_DGF_charge_discharge_stochastic(df,
                              node="Germany",
                              period="2045-2050",
                              gasscenario=1,
                              scenario='scenario1')


    df = pd.read_csv(result_dir / "results_hydrogen_storage_operational.csv")
    plot_discharge_cycles_sawtooth(
        df,
        node="Germany",
        period="2045-2050",
        gasscenario=1,
        tech="DGF",
        capacity=6545.9,
        scenario='scenario1'
    )

    plot_power_balance_for_high_h2(Power_balance,'2045-2050','scenario1','Germany')

    plot_storage_charge_discharge_total(df,'2045-2050',1,'scenario1','DGF')
    plot_storage_charge_discharge_total(df, '2045-2050', 1, 'scenario1', 'Aquifer')
    plot_storage_charge_discharge_total(df, '2045-2050', 1, 'scenario1', 'SaltCavern')


    merged = plot_h2_prod_discharge_plus_demand(
        hydrogen_production,
        df,
        hydrogen_use,
        period="2050-2055",
        gasscenario=1,
        scenario="scenario2",  # eller None for snitt
    )









