We provide a list of combinations of models, tasks and hyperparameters that do not yield useful results. Using the hyperparameters from [Niklaus et al. (2023)](https://arxiv.org/abs/2301.13126) the following constellations yield either eval loss == null and/or predict losss = null.

| finetuning_task                                      | \_name_or_path                   | language | missing_seeds | revision                                 |
| ---------------------------------------------------- | -------------------------------- | -------- | ------------- | ---------------------------------------- |
| swiss_criticality_prediction_bge_facts               | joelito/legal-swiss-roberta-base | all      | 2             | 47048d047356032a7f3e322469744294a5923042 |
| swiss_criticality_prediction_bge_considerations      | joelito/legal-swiss-roberta-base | all      | 2             | 47048d047356032a7f3e322469744294a5923042 |
| swiss_criticality_prediction_citation_facts          | ZurichNLP/swissbert-xlm-vocab    | all      | 3             | main                                     |
| swiss_criticality_prediction_citation_facts          | facebook/xmod-base               | all      | 2             | main                                     |
| swiss_criticality_prediction_citation_considerations | joelito/legal-swiss-roberta-base | all      | 3             | 47048d047356032a7f3e322469744294a5923042 |
| swiss_law_area_prediction_sub_area_facts             | facebook/xmod-base               | all      | 1             | main                                     |
| swiss_law_area_prediction_sub_area_considerations    | facebook/xmod-base               | all      | 2             | main                                     |
