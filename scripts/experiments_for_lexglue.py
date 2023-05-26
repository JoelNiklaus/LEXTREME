import pandas as pd
import os


tasks = [
    "eurlex",
    "scotus",
    "ledgar",
    "unfair_tos",
    "case_hold",
    "ecthr_a",
]

missing_runs = pd.read_excel('../utils/results/lextreme/paper_results/report.xlsx', sheet_name='completeness_report')
missing_runs = missing_runs[missing_runs.finetuning_task.isin(tasks)]
missing_runs = missing_runs[missing_runs._name_or_path.str.contains('joelito.*large')]

print(missing_runs.shape)
print(missing_runs.head())

os.chdir('../')

for row in missing_runs.to_dict(orient="records"):
    name_or_path = row["_name_or_path"]
    revision = row["revision"]
    ft = row['finetuning_task']
    seeds = row['missing_seeds']
    command = 'python main.py -gn 0,1 -gm 80 -los ' + seeds + ' --lower_case true -bz 4 -as 2 -mfbm micro-f1 -nte 20 -lr 3e-5 -esp 3 -ld paper_results ' + '-t ' + ft + ' -lmt ' + name_or_path + ' -rev ' + revision
    os.system(command)
