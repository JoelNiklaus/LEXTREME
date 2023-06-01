import pandas as pd
import os

tasks = [
    "scotus"
]

missing_runs = pd.read_excel(
    '../utils/results/lextreme/paper_results/report.xlsx', sheet_name='completeness_report')
missing_runs = missing_runs[missing_runs.finetuning_task.isin(tasks)]
missing_runs = missing_runs[missing_runs._name_or_path.str.contains(
    'joelito.*large')]
missing_models = list(missing_runs._name_or_path.unique())
missing_tasks = list(missing_runs.finetuning_task.unique())
missing_runs_new = list()

revision_lookup = dict()
for row in missing_runs.to_dict(orient="records"):
    revision_lookup[row['_name_or_path']]=row['revision']

for row in missing_runs.to_dict(orient="records"):
    for s in row["missing_seeds"].split(','):
        for mm in missing_models:
            for task in missing_tasks:
                entry = dict()
                entry["_name_or_path"] = mm
                entry["revision"] = revision_lookup[mm]
                entry['finetuning_task'] = task
                entry["missing_seeds"] = s
                missing_runs_new.append(entry)

missing_runs_new = pd.DataFrame(missing_runs_new)
missing_runs_new = missing_runs_new.drop_duplicates(
    ['_name_or_path', 'missing_seeds', 'finetuning_task'])

print(missing_runs_new.shape)
print(missing_runs_new.head())

os.chdir('../')

for row in missing_runs_new.to_dict(orient="records"):
    name_or_path = row["_name_or_path"]
    revision = row["revision"]
    ft = row['finetuning_task']
    seeds = row['missing_seeds']
    command = 'python main.py -gn 0_1 -gm 80 -los ' + seeds + \
        ' --lower_case true -bz 4 -as 2 -mfbm micro-f1 -nte 20 -lr 1e-5 -esp 3 -ld paper_results --weight_decay 0.06 --warmup_ratio 0.1 ' + \
        '-t ' + ft + ' -lmt ' + name_or_path + ' -rev ' + revision
    os.system(command)
