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
                            plot_storage_charge_discharge_total)


project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir = data_dir / "Results_BASE_BASE" / "full_model_base"
plot_dir = data_dir / "Results_BASE_BASE" #lagrer figurene i resultat mappen
plot_dir.mkdir(exist_ok=True)

Lagre_figurer =False
figurnavn = "BASE_final"

gen_info='yes'
H2_prod='yes'
el_demand='yes'
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
    Yearly_hydrogenProd_perTech(hydrogen_production, 'PEM_green production [ton]', 'PEM_yellow production [ton]',
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

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def plot_h2_prod_discharge_plus_demand(
            df_prod,  # produksjons-DF
            df_store,  # lager-DF (kun *_discharge [ton] trengs)
            df_demand,  # demand-DF (med sektorkolonnene)
            period="2020-2025",
            gasscenario=1,
            scenario=None,  # None => gjennomsnitt over scenarioer
            nodes=None,  # None => alle noder
            prod_tech_cols=None,  # prod-kolonner
            storage_discharge_cols=None,
            # f.eks. ["SaltCavern_discharge [ton]","DGF_discharge [ton]","Aquifer_discharge [ton]"]
            demand_cols=None,
            # f.eks. ["Hydrogen used for steel [ton]","Hydrogen used for cement [ton]","Hydrogen used for ammonia [ton]","Hydrogen used for oil refining [ton]","Hydrogen used for transport [ton]"]
            title_prefix="H2 production + storage discharge (stacked) with total demand (line)"
    ):
        # Standardprodusenter
        default_prod_cols = [
            "ALK production [ton]",
            "SOEC production [ton]",
            "Reformer production [ton]",
            "PEM_yellow production [ton]",
            "PEM_green production [ton]",
        ]
        prod_tech_cols = prod_tech_cols or default_prod_cols

        # Gjetting av kolonner
        if storage_discharge_cols is None:
            storage_discharge_cols = [c for c in df_store.columns if c.endswith("_discharge [ton]")]
        if demand_cols is None:
            demand_cols = [
                "Hydrogen used for steel [ton]",
                "Hydrogen used for cement [ton]",
                "Hydrogen used for oil refining [ton]",
                "Hydrogen used for ammonia [ton]",
                "Hydrogen used for transport [ton]",
            ]

        # Påkrevde meta-kolonner
        req = ["Node", "Period", "Scenario", "GasScenario", "Season", "Hour"]

        # Valider kolonner
        miss_p = [c for c in (req + prod_tech_cols) if c not in df_prod.columns]
        miss_s = [c for c in (req + storage_discharge_cols) if c not in df_store.columns]
        miss_d = [c for c in (req + demand_cols) if c not in df_demand.columns]
        if miss_p: raise ValueError(f"Mangler kolonner i produksjons-DF: {miss_p}")
        if miss_s: raise ValueError(f"Mangler kolonner i lager-DF: {miss_s}")
        if miss_d: raise ValueError(f"Mangler kolonner i demand-DF: {miss_d}")

        # Numerisk casting
        for c in prod_tech_cols: df_prod[c] = pd.to_numeric(df_prod[c], errors="coerce").fillna(0.0)
        for c in storage_discharge_cols: df_store[c] = pd.to_numeric(df_store[c], errors="coerce").fillna(0.0)
        for c in demand_cols: df_demand[c] = pd.to_numeric(df_demand[c], errors="coerce").fillna(0.0)

        # Filtreringshjelper
        def _mask(df):
            m = (df["Period"] == period) & (df["GasScenario"] == gasscenario)
            if scenario is not None: m &= (df["Scenario"] == scenario)
            if nodes is not None:
                nodes_set = set(nodes) if isinstance(nodes, (list, tuple, set)) else {nodes}
                m &= df["Node"].isin(nodes_set)
            return m

        psub = df_prod.loc[_mask(df_prod), req + prod_tech_cols].copy()
        ssub = df_store.loc[_mask(df_store), req + storage_discharge_cols].copy()
        dsub = df_demand.loc[_mask(df_demand), req + demand_cols].copy()
        if psub.empty: raise ValueError("Ingen produksjonsrader som matcher filteret.")
        if ssub.empty: raise ValueError("Ingen lagerrader som matcher filteret.")
        if dsub.empty: raise ValueError("Ingen demand-rader som matcher filteret.")

        # SUM over noder per (Scenario,Season,Hour)
        p_by = psub.groupby(["Scenario", "Season", "Hour"], as_index=False)[prod_tech_cols].sum()
        s_by = ssub.groupby(["Scenario", "Season", "Hour"], as_index=False)[storage_discharge_cols].sum()
        d_by = dsub.groupby(["Scenario", "Season", "Hour"], as_index=False)[demand_cols].sum()

        # Forventning over scenarioer (eller beholde valgt)
        if scenario is None:
            p_grp = p_by.groupby(["Season", "Hour"], as_index=False)[prod_tech_cols].mean(numeric_only=True)
            s_grp = s_by.groupby(["Season", "Hour"], as_index=False)[storage_discharge_cols].mean(numeric_only=True)
            d_grp = d_by.groupby(["Season", "Hour"], as_index=False)[demand_cols].mean(numeric_only=True)
            nodes_lbl = "ALL" if nodes is None else ",".join(
                sorted(set(nodes if isinstance(nodes, (list, tuple, set)) else [nodes])))
            suffix = f"(avg over scenarios; nodes={nodes_lbl})"
        else:
            p_grp = p_by.groupby(["Season", "Hour"], as_index=False)[prod_tech_cols].sum(numeric_only=True)
            s_grp = s_by.groupby(["Season", "Hour"], as_index=False)[storage_discharge_cols].sum(numeric_only=True)
            d_grp = d_by.groupby(["Season", "Hour"], as_index=False)[demand_cols].sum(numeric_only=True)
            nodes_lbl = "ALL" if nodes is None else ",".join(
                sorted(set(nodes if isinstance(nodes, (list, tuple, set)) else [nodes])))
            suffix = f"(scenario {scenario}; nodes={nodes_lbl})"

        # Sortér og tidsakse
        season_order = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3, "fall": 3}
        for df_ in (p_grp, s_grp, d_grp):
            df_["season_order"] = df_["Season"].map(season_order).fillna(99)
            df_.sort_values(["season_order", "Hour"], inplace=True)
            df_.reset_index(drop=True, inplace=True)
            df_["t"] = np.arange(len(df_))

        # Merge på felles (Season,Hour,t)
        merged = p_grp[["Season", "Hour", "t"] + prod_tech_cols] \
            .merge(s_grp[["Season", "Hour", "t"] + storage_discharge_cols], on=["Season", "Hour", "t"], how="inner") \
            .merge(d_grp[["Season", "Hour", "t"] + demand_cols], on=["Season", "Hour", "t"], how="inner")

        # Total demand (linje)
        # Total demand (linje)
        merged["H2 demand total [ton/h]"] = merged[demand_cols].sum(axis=1)

        # --- STACK-SERIER ---

        # Produksjon + discharge (positiv stack)
        prod_stack_series = [merged[c].to_numpy() for c in prod_tech_cols]
        prod_stack_labels = prod_tech_cols

        store_stack_series = [merged[c].to_numpy() for c in storage_discharge_cols]
        store_stack_labels = [c.replace("_discharge [ton]", " discharge [ton]") for c in storage_discharge_cols]

        # Demand (negativ stack – går nedover fra 0)
        demand_stack_series = [-merged[c].to_numpy() for c in demand_cols]
        demand_stack_labels = [c.replace("Hydrogen used for ", "").replace(" [ton]", "") for c in demand_cols]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Positiv stack: produksjon + lagertømming
        ax.stackplot(
            merged["t"],
            prod_stack_series + store_stack_series,
            labels=prod_stack_labels + store_stack_labels,
            alpha=0.9,
        )

        # Negativ stack: ulike demand-komponenter
        ax.stackplot(
            merged["t"],
            demand_stack_series,
            labels=demand_stack_labels,
            alpha=0.9,
        )

        # Total demand som linje (valgfritt, men ofte greit å beholde)
        ax.plot(
            merged["t"],
            merged["H2 demand total [ton/h]"] * -1,  # også negativ, så den følger stacken under 0
            label="Total hydrogen demand [ton/h]",
            linewidth=1.8,
        )

        # Null-linje
        ax.axhline(0, linewidth=0.8)

        ax.set_xlabel("Hour (ordered by Season → Hour)")
        ax.set_ylabel("Hydrogen [ton/h]")
        ax.set_title(f"{title_prefix} {suffix}")

        # Legend: samle alle labels fra begge stackplots + linje
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", ncol=2, fontsize=8, frameon=False)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return merged[["Season", "Hour", "t"]
                      + prod_tech_cols
                      + storage_discharge_cols
                      + demand_cols
                      + ["H2 demand total [ton/h]"]]


    merged = plot_h2_prod_discharge_plus_demand(
        hydrogen_production,
        df,
        hydrogen_use,
        period="2050-2055",
        gasscenario=1,
        scenario="scenario2",  # eller None for snitt
    )









