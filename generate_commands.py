import pandas as pd
import argparse
import os

lex_glue_tasks = [
    "eurlex",
    "scotus",
    "ledgar",
    "unfair_tos",
    "case_hold",
    "ecthr_a",
    "ecthr_b"
]

df = pd.read_excel('utils/results/lextreme/paper_results/report.xlsx')
# df = df[df.run == "x"]
df = df[df._name_or_path.str.contains("longform")]
df = df[df.finetuning_task.isin(lex_glue_tasks) == False]


def generate_commands(gm, gn, ld):
    commands = list()
    for r in df.to_dict(orient="records"):
        comm = 'python main.py -gm ' + str(gm) + ' -gn ' + str(gn) + ' -t ' + r["finetuning_task"] + ' -lmt ' + \
               r["_name_or_path"] + ' -los ' + r['missing_seeds'] + \
               ' -ld ' + ld + ' --revision ' + r['revision']
        commands.append(comm)

    return ' ; sleep 10 ; '.join(commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-gn')
    parser.add_argument('-gm')
    parser.add_argument('-ld')

    args = parser.parse_args()

    commands = generate_commands(args.gm, args.gn, args.ld)

    os.system(commands)
