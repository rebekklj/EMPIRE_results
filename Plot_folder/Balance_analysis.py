from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir1 = data_dir / "Results_BalanceC8_trans" / "full_model_base"
result_dir2 = data_dir / 'Results_BalanceCosts_trans' / 'full_model_base'
result_dir3 = data_dir / 'Results_balanceC55_trans' / 'full_model_base'
result_dir4 = data_dir / 'Results_balanceC2_5' / 'full_model_base'

# In[] Fornybar energi (variabel) produksjon VS balansekostnader

VRES=['Windonshore']

Elec_gentype1=pd.read_csv(result_dir1 / 'results_elec_generation_inv.csv')
Elec_gentype2=pd.read_csv(result_dir2 / 'results_elec_generation_inv.csv')
Elec_gentype3=pd.read_csv(result_dir3 / 'results_elec_generation_inv.csv')
Elec_gentype4 = pd.read_csv(result_dir4 / 'results_elec_generation_inv.csv')
def Balance_analysis(Elec_gentype,VRES):
    TotalGen_perTech = (
            Elec_gentype
            .groupby(['GeneratorType'], as_index=False)['genExpectedAnnualProduction_GWh']
            .sum()
        )
    TotalGen_perTech['genExpectedAnnualProduction_GWh']=TotalGen_perTech['genExpectedAnnualProduction_GWh']*5/1000


    VRES_prod=0 #GWh
    Nuc_prod=TotalGen_perTech.loc[TotalGen_perTech['GeneratorType']=='Nuclear','genExpectedAnnualProduction_GWh'].iloc[0]
    Other=0

    for i in VRES:
        VRES_prod+=TotalGen_perTech.loc[TotalGen_perTech['GeneratorType']==i,'genExpectedAnnualProduction_GWh'].iloc[0]

    for i in TotalGen_perTech['GeneratorType']:
        Other+=TotalGen_perTech.loc[TotalGen_perTech['GeneratorType']==i,'genExpectedAnnualProduction_GWh'].iloc[0]
    Other=Other-VRES_prod-Nuc_prod

    return VRES_prod,Nuc_prod,Other

VRES1,Nuc1,Other1=Balance_analysis(Elec_gentype1,VRES)
VRES2,Nuc2,Other2=Balance_analysis(Elec_gentype2,VRES)
VRES3,Nuc3,Other3=Balance_analysis(Elec_gentype3,VRES)
VRES4,Nuc4,Other4 = Balance_analysis(Elec_gentype4,VRES)

# Data
x = np.array([2.55,4.67, 8.49, 16.98])
VRES_list = np.array([VRES4,VRES3, VRES1, VRES2])
Nuc_list  = np.array([Nuc4,Nuc3,  Nuc1,  Nuc2])
Other_list= np.array([Other4,Other3,Other1,Other2])

# Lineær fit (y = a x + b)
a_vres,  b_vres  = np.polyfit(x, VRES_list, 1)
a_nuc,   b_nuc   = np.polyfit(x, Nuc_list,  1)
a_other, b_other = np.polyfit(x, Other_list,1)

# Tett x-akse for glatte linjer
xfit = np.linspace(x.min(), x.max(), 200)

# Beregn linjene
yfit_vres  = a_vres*xfit  + b_vres
yfit_nuc   = a_nuc*xfit   + b_nuc
yfit_other = a_other*xfit + b_other

# Plot datapunkter
plt.figure(figsize=(10,6))
plt.plot(x, VRES_list,  marker='o',markersize='8', color='thistle',  label=f'VRES',linewidth=3)
plt.plot(x, Nuc_list,    marker='o',markersize='8', color='steelblue',   label=f'Nuclear',linewidth=3)
plt.plot(x, Other_list,  marker='o',markersize='8', color='darkseagreen', label=f'Other',linewidth=3)

# Plot regresjonslinjer
plt.plot(xfit, yfit_vres,  color='thistle',linestyle='--',  label=f'a={a_vres:.3f}')
plt.plot(xfit, yfit_nuc,   color='steelblue',linestyle='--',   label=f'a={a_nuc:.3f}')
plt.plot(xfit, yfit_other, color='darkseagreen',linestyle='--', label=f'a={a_other:.3f}')

plt.xlabel('Balance and reserve costs for VRES [EUR/MWh]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Total power production [TWh]',fontsize=14)
plt.ylim(50000,92000)
plt.grid(True)
plt.legend(loc='upper right',title='Energy source', fontsize=12,title_fontsize=14)
plt.tight_layout()
plt.savefig(data_dir/'Balance_costs_impact.eps', format='eps', bbox_inches='tight')
plt.show()









