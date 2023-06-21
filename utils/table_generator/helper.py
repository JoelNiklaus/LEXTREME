from typing import List, Dict, AnyStr, Union, Literal, Optional, Set
import numpy as np
import os
import pandas as pd
import sys
from copy import deepcopy

TABLE_TO_REPORT_SPECS = os.path.join(os.path.dirname(__file__), '../report_specs')
TABLE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../tables')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utilities import get_meta_infos

meta_infos = get_meta_infos()

MONOLINGUAL = 'monolingual'
MULTILINGUAL = 'multilingual'

LATEXT_TABLE_PREFIX = "\\begin{table*}[!ht]\n\\centering\n"
LATEXT_TABLE_SUFFIX = "\\caption{Table explanation text.}\n\\label{tab:table_label}\n\\end{table*}"


def insert_responsibilities(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    response = pd.read_excel(os.path.join(
        os.path.dirname(__file__), 'run_responsibility.xlsx'))
    response = response[['finetuning_task', '_name_or_path', 'responsible', 'gpu']]
    df_merged = pd.merge(dataframe, response, how='left', right_on=[
        'finetuning_task', '_name_or_path'], left_on=['finetuning_task', '_name_or_path'])
    return df_merged


def insert_abbreviations_in_columns(dataframe):
    columns = dataframe.columns.tolist()
    for c in columns:
        if c in meta_infos['task_abbreviations'].keys():
            dataframe.rename(
                columns={c: meta_infos['task_abbreviations'][c]}, inplace=True)
        elif c in meta_infos['dataset_abbreviations'].keys():
            dataframe.rename(
                columns={c: meta_infos['dataset_abbreviations'][c]}, inplace=True)

    return dataframe


def round_value(value: float, places=1):
    if isinstance(value, float):
        return round(value * 100, places)
    elif isinstance(value, tuple) and isinstance(value[0], float) and isinstance(value[1], float):
        # This is needed for cases where we insert standard deviation in the cells
        score = round(value[0] * 100, places)
        standard_deviation = round(value[1] * 100, places)
        return (score, standard_deviation)
    else:
        return value


def replace_empty_string(value, placeholder):
    if value == '':
        return placeholder
    else:
        return value


def make_highest_value_latex_bold(values: List[Union[int, float]]) -> list:
    """
    Finds the highest value in a list of numbers and turns the value into a string and adds latext tag to make it appear bold in latex.

    @param values: List of integers or floats.
    @return: List.
    """
    values_to_check = [v if isinstance(v, float) else 0 for v in values]
    highest_value_index = np.argmax(values_to_check)
    values[highest_value_index] = '\\bf{' + str(values[highest_value_index]) + '}'
    return values


def highligh_highest_value(dataframe: pd.core.frame.DataFrame, column: str) -> pd.core.frame.DataFrame:
    values = dataframe[column].tolist()
    # if len([v for v in values if type(literal_eval(v))!=str])==len(values):
    values = make_highest_value_latex_bold(values)
    dataframe[column] = values
    return dataframe


def make_latext_table(dataframe: pd.core.frame.DataFrame, file_name: str):
    dataframe.fillna('', inplace=True)
    dataframe = dataframe.applymap(
        lambda x: replace_empty_string(x, '-'))
    available_columns = dataframe.columns.tolist()
    for col in available_columns:
        dataframe = highligh_highest_value(dataframe=dataframe, column=col)
        dataframe.rename(columns={col: '\\bf{' + col + '}'}, inplace=True)

    with open(file_name, 'w') as f:
        print(LATEXT_TABLE_PREFIX + dataframe.to_latex(index=True,
                                                       escape=False,
                                                       column_format='r' * len(
                                                           dataframe.columns.tolist())) + LATEXT_TABLE_SUFFIX, file=f)


def remove_outliers(list_of_values: list):
    removed_values = list()
    list_of_values_copy = list()
    if len(list_of_values) >= 3:
        standard_deviation = np.std(list_of_values)
        for value in list_of_values:
            if value < standard_deviation:
                removed_values.append(value)
            else:
                list_of_values_copy.append(value)
    if len(list_of_values_copy) == 0:
        list_of_values_copy = deepcopy(list_of_values)
        removed_values = []
    return list_of_values_copy, removed_values
