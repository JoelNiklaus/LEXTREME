# What does the code do

The python script `create_overview.py` is used to process and clear the results of the runs logged in [wandb](https://wandb.ai/site) and to create overviews of the scores (F1, accuracy) that are used in the paper.

# Folders

The directory contains the following folders:

- logs: Everytime you run the code, log messages will be generated that are stored in specific log files. You can use them for debugging.
- results: All result files with the (aggregated) scores are stored in this directory

# How to use the code

You can have a look at the notebook [Documentation_data_analysis](https://github.com/JoelNiklaus/LEXTREME/blob/main/utils/Documentation_data_analysis.ipynb)

You can also just run `python create_overview.py` and it will automatically create a report saved in an excel filed called `report.xlsx` in the folder `results`. If you want to consider only specific finetuning tasks, you can add these in the json file called `tasks_for_report.json`.
