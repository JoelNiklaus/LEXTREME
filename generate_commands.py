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

lextreme_tasks = ["brazilian_court_decisions_judgment", "brazilian_court_decisions_unanimity", "german_argument_mining",
                  "greek_legal_code_chapter", "greek_legal_code_subject", "greek_legal_code_volume",
                  "swiss_judgment_prediction", "online_terms_of_service_unfairness_levels",
                  "online_terms_of_service_clause_topics", "covid19_emergency_event", "multi_eurlex_level_1",
                  "multi_eurlex_level_2", "multi_eurlex_level_3", "greek_legal_ner", "legalnero", "lener_br",
                  "mapa_coarse", "mapa_fine"
                  ]

allowed_finetuning_tasks = []
excluded_finetuning_tasks = []
allowed_name_or_path = []
excluded_name_or_path = []



def generate_commands(gm: str, gn: str, ld: str, rs: str) -> None:
    """

    @param gm: gpu memory
    @param gn: gpu numbers
    @param ld: logging directory
    @param rs: report_specs
    @return:
    """
    os.system(' python utils/table_generator/create_overview.py --rs ' + rs)

    df = pd.read_excel(f'utils/tables/lextreme/{rs}/report.xlsx')

    if len(allowed_finetuning_tasks) > 0:
        df = df[df.finetuning_task.isin(allowed_finetuning_tasks)]
    if len(excluded_finetuning_tasks) > 0:
        df = df[df.finetuning_task.isin(excluded_finetuning_tasks) == False]
    if len(allowed_name_or_path) > 0:
        df = df[df._name_or_path.isin(allowed_name_or_path)]
    if len(excluded_name_or_path) > 0:
        df = df[df._name_or_path.isin(excluded_name_or_path) == False]

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
    parser.add_argument('-rs')

    args = parser.parse_args()

    commands = generate_commands(args.gm, args.gn, args.ld)

    os.system(commands)
