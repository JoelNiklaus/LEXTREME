import os
import pandas as pd


TABLE_TO_REPORT_SPECS = os.path.join(os.path.dirname(__file__), '../report_specs')
TABLE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../tables')
MONOLINGUAL = 'monolingual'
MULTILINGUAL = 'multilingual'


def insert_responsibilities(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    response = pd.read_excel(os.path.join(
        os.path.dirname(__file__), 'run_responsibility.xlsx'))
    response = response[['finetuning_task', '_name_or_path', 'responsible', 'gpu']]
    df_merged = pd.merge(dataframe, response, how='left', right_on=[
        'finetuning_task', '_name_or_path'], left_on=['finetuning_task', '_name_or_path'])
    return df_merged
