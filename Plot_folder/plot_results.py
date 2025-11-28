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
                            plot_discharge_cycles_sawtooth,plot_power_balance_for_high_h2)


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_BASE" / "full_model_base"
plot_dir = data_dir / "Results_BASE" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer =False
figurnavn = "BASE"

gen_info='no'
H2_prod='yes'
el_demand='no'
H2_storage='no'

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
if H2_prod=='yes':
    hydrogen_production=pd.read_csv(result_dir/ 'results_hydrogen_production.csv')
    HydrogenProd_piechart(hydrogen_production, 12, 2, savefigure=Lagre_figurer, figurename=figurnavn, results_dir=plot_dir)

# In[]

if el_demand=='yes':
    Power_balance['Power reformer plant [MWh]'] = Power_balance['Power reformer plant [MWh]'] * (-1)
    plot_power_demand(Power_balance,n_hours=12)

# In[]
if H2_prod=='yes':
    hydrogen_use=pd.read_csv(result_dir/ 'results_hydrogen_use.csv')
    plot_hydrogen_use(hydrogen_use, 12, 2, savefigure=False, figurename=None, results_dir=None)

    import matplotlib.pyplot as plt
    import pandas as pd

    import matplotlib.pyplot as plt
    import pandas as pd


    def hydrogen_prod_vs_import_bar(df):
        # --- Beregn produksjon, import, eksport ---

        h2_prod_node = (
                df.groupby("Node")["Hydrogen produced [ton]"].sum() / (2 * 10 ** 6)
        )

        h2_import_node = (
                df.groupby("Node")["Hydrogen imported pipe [ton]"].sum() / (2 * 10 ** 6)
        )

        h2_export_node = -(
                df.groupby("Node")["Hydrogen exported pipe [ton]"].sum() / (2 * 10 ** 6)
        )

        # Dataframe for plotting
        df_plot = pd.DataFrame({
            "Produced": h2_prod_node,
            "Imported": h2_import_node,
            "Exported": h2_export_node
        }).fillna(0)

        # --- Sortér etter total supply (Produced + Imported) ---

        df_plot["TotalSupply"] = df_plot["Produced"] + df_plot["Imported"]
        df_plot = df_plot.sort_values("TotalSupply", ascending=False)
        df_plot = df_plot.drop(columns="TotalSupply")

        # --- Plotting ---

        fig, ax = plt.subplots(figsize=(12, 6))

        nodes = df_plot.index
        x = range(len(nodes))
        width = 0.35

        # Supply bar (Produced + Imported)
        ax.bar(
            [i - width / 2 for i in x],
            df_plot["Produced"],
            width=width,
            label="Produced",
            color="#1f77b4"
        )
        ax.bar(
            [i - width / 2 for i in x],
            df_plot["Imported"],
            width=width,
            bottom=df_plot["Produced"],
            label="Imported",
            color="#aec7e8"
        )

        # Outflow bar (Exported)
        ax.bar(
            [i + width / 2 for i in x],
            df_plot["Exported"],
            width=width,
            label="Exported",
            color="#ffbb78"
        )

        # Layout
        ax.set_xticks(list(x))
        ax.set_xticklabels(nodes, rotation=90,fontsize=14)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel("Hydrogen [million ton]",fontsize=18)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=18)

        plt.tight_layout()
        plt.show()

        return df_plot


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
                              node="Italy",
                              period="2050-2055",
                              gasscenario=1,
                              scenario='scenario1')



    plot_discharge_cycles_sawtooth(
        df,
        node="Italy",
        period="2050-2055",
        gasscenario=1,
        tech="DGF",
        capacity=6545.9,
        scenario='scenario1'
    )



    plot_power_balance_for_high_h2(Power_balance,'2050-2055','scenario1','Italy')














