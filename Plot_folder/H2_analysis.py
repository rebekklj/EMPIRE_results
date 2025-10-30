from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions_plot import transmission_map

project_dir = Path(__file__).resolve().parents[1] #EMPIRE_results_git mappen
data_dir = project_dir / "data"
result_dir1 = data_dir / "Results_PEM_types2" / "full_model_base"

H2_sent=pd.read_csv(result_dir1 / "results_hydrogen_pipeline_inv.csv")


transmission_map(H2_sent,'Pipeline capacity built [ton/hr]','lightseagreen',12,2, savefigure=False, figurename='new_CAPEX_H2_built', results_dir=result_dir1)


transmission_map(H2_sent,'Pipeline capacity repurposed [ton/hr]','darkgreen',12,2, savefigure=False, figurename='new_CAPEX_H2_repu', results_dir=result_dir1)

print(H2_sent['Pipeline capacity repurposed [ton/hr]'].min())

