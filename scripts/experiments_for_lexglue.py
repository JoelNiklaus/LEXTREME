import pandas as pd
import os

path_to_file = os.path.join(os.path.dirname(__file__), '../utils/joels_models_revision_infos_MultiLegalPile.xlsx')
models = pd.read_excel(path_to_file)

models = models[models.need_to_be_run_with_LexGlue == True]._name_or_path.tolist()

tasks = [
    "eurlex",
    "scotus",
    "ledgar",
    "unfair_tos",
    "case_hold"
  ]



missing_runs = pd.read_excel('../utils/results/report.xlsx', sheet_name = 'completeness_report')
missing_runs = missing_runs[missing_runs.finetuning_task.isin(tasks)]
missing_runs = missing_runs[missing_runs._name_or_path.isin(models)]

print(missing_runs.shape)
print(missing_runs.head())

os.chdir('../')


for row in missing_runs.to_dict(orient="records"):
    name_or_path = row["_name_or_path"]
    revision = row["revision"]
    ft = row['finetuning_task']
    command = 'python main.py -gn 0 -gm 80 -los 1,2,3,4,5 --lower_case true -bz 4 -as 2 -mfbm micro-f1 -nte 20 -lr 3e-5 -esp 3 -ld paper_results ' + '-t '+ft +' -lmt '+ name_or_path +' -rev '+revision
    os.system(command)
