import pandas as pd
import os

path_to_file = os.path.join(os.path.dirname(__file__), '../utils/joels_models_revision_infos_MultiLegalPile.xlsx')
models = pd.read_excel(path_to_file)

models = models[models.need_to_be_run_with_LexGlue == True]

tasks = [
    "ecthr_a",
    "ecthr_b",
    "eurlex",
    "scotus",
    "ledgar",
    "unfair_tos",
    "case_hold"
  ]

os.chdir('../')

for row in models.to_dict(orient="records"):
    name_or_path = row["_name_or_path"]
    revision = row["revision"]
    for ft in tasks:
        command = 'python main.py -gn 0 -gm 80 -los 1,2,3,4,5 --lower_case true -bz 8 -mfbm micro-f1 -nte 20 -lr 3e-5 -esp 3 -ld paper_results ' + '-t '+ft
        os.system(command)
