# LEXTREME: A Multi-Lingual Benchmark Dataset for Legal Language Understanding

## Setup

It works best with python 3.9 and torch==1.10.0+cu113. Otherwise, we experienced problems with fp16 training and evaluation.

```bash
# install torch like this to avoid fp16 problems
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

In case you get the error `AttributeError: module 'distutils' has no attribute 'version'` (https://github.com/pytorch/pytorch/issues/69894)
Run

```bash
pip install setuptools==59.5.0
```

In order to log the training results, we used [Weights & Biases](https://wandb.ai/site/). When running the script below, you will be asked if you want to use Weights & Biases or not. In case you want to use Weights & Biases too, you should log in to your Weights & Biases account beforehand, by typing the following command in the terminal:

```
wandb login {WANDB_API_KEY}
```

You can find WANDB_API_KEY in your profile setting on [Weights & Biases](https://wandb.ai/site/) after signing up or login.

## Dataset Summary

[LEXTREME](https://huggingface.co/datasets/joelito/lextreme) consist of three classification task types:

- Single Label Text Classification (SLTC)
- Multi Label Text Classification (MLTC)
- Named Entity Recognition (NER)

The dataset consists of 11 diverse multilingual legal NLU (natural language understanding) datasets. Six datasets have one single configuration and five datasets have two or three configurations. This leads to a total of 18 tasks (8 SLTC, 5 MLTC and 5 NER).

We use the existing train, validation, and test splits if present. In the other cases we split the data ourselves (80\% train, 10\% validation and test each).

## Supported Tasks

For a detailed description of each task and dataset, see [Niklaus et al. (2023)](https://arxiv.org/abs/2301.13126). Datasets are abbreviated by three capital letters. Configurations of datasets, in case they exist, are indicated by an additional letter separated by a hyphen.

| Task                                               | Type                             | Train / Dev / Test Examples | Train / Dev / Test Labels |
| -------------------------------------------------- | -------------------------------- | --------------------------- | ------------------------- |
| BCD-J (brazilian_court_decisions_judgment)         | SLTC (Judgment Prediction)       | 3234 / 404 / 405            | 3 / 3 / 3                 |
| BCD-U (brazilian_court_decisions_unan)             | SLTC (Judgment Prediction)       | 1715 / 211 / 204            | 2 / 2 / 2                 |
| GAM (german_argument_mining)                       | SLTC (Argument Mining)           | 19271 / 2726 / 3078         | 4 / 4 / 4                 |
| GLC-V (greek_legal_code_volume)                    | SLTC (Topic Classification)      | 28536 / 9511 / 9516         | 47 / 47 / 47              |
| GLC-C (greek_legal_code_chapter)                   | SLTC (Topic Classification)      | 28536 / 9511 / 9516         | 386 / 377 / 374           |
| GLC-S (greek_legal_code_subject)                   | SLTC (Topic Classification)      | 28536 / 9511 / 9516         | 2143 / 1679 / 1685        |
| SJP (swiss_judgment_prediction)                    | SLTC (Judgment Prediction)       | 59709 / 8208 / 17357        | 2 / 2 / 2                 |
| OTS-UL (online_terms_of_service_unfairness_levels) | SLTC (Unfairness Classification) | 2074 / 191 / 417            | 3 / 3 / 3                 |
| OTS-CT (online_terms_of_service_clause_topics)     | MLTC (Unfairness Classification) | 19942 / 1690 / 4297         | 9 / 8 / 9                 |
| C19 (covid19_emergency_event)                      | MLTC (Event Classification)      | 3312 / 418 / 418            | 8 / 8 / 8                 |
| MEU-1 (multi_eurlex_level_1)                       | MLTC (Topic Classification)      | 817239 / 112500 / 115000    | 21 / 21 / 21              |
| MEU-2 (multi_eurlex_level_2)                       | MLTC (Topic Classification)      | 817239 / 112500 / 115000    | 127 / 126 / 127           |
| MEU-3 (multi_eurlex_level_3)                       | MLTC (Topic Classification)      | 817239 / 112500 / 115000    | 500 / 454 / 465           |
| GLN (greek_legal_ner)                              | NER                              | 17699 / 4909 / 4017         | 17 / 17 / 17              |
| LNR (legalnero)                                    | NER                              | 7552 / 966 / 907            | 11 / 9 / 11               |
| LNB (lener_br)                                     | NER                              | 7828 / 1177 / 1390          | 13 / 13 / 13              |
| MAP-C (mapa_coarse)                                | NER                              | 27823 / 3354 / 10590        | 13 / 11 / 11              |
| MAP-F (mapa_fine)                                  | NER                              | 27823 / 3354 / 10590        | 44 / 26 / 34              |

## LEXTREME Scores

We evaluated multilingual models as well as monolingual models. The multilingual models are the following:

| **Model**                                                               | **Source**                                                                                             | **Parameters** | **Vocabulary Size** | **Pretraining Specs** | **Pretraining Corpora** | **Pretraining Languages** |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------- | ------------------- | --------------------- | ----------------------- | ------------------------- |
| [MiniLM](microsoft/Multilingual-MiniLM-L12-H384)                        | [Wang et al. (2020)](https://dl.acm.org/doi/pdf/10.5555/3495724.3496209)                               | 118M           | 250K                | 1M steps / BS 256     | 2.5T CC100 data         | 100                       |
| [DistilBert](https://huggingface.co/distilbert-base-multilingual-cased) | [Sanh (2019)](https://arxiv.org/pdf/1910.01108.pdf)                                                    | 135M           | 120K                | BS up to 4000         | Wikipedia               | 104                       |
| [mDeberta-v3](https://huggingface.co/microsoft/mdeberta-v3-base)        | He et al. ([2020](https://arxiv.org/pdf/2006.03654.pdf), [2021](https://arxiv.org/pdf/2111.09543.pdf)) | 278M           | 128K                | 500K steps / BS 8192  | 2.5T CC100 data         | 100                       |
| [XLM-R base](https://huggingface.co/xlm-roberta-base)                   | [Conneau et al. (2020)](https://aclanthology.org/2020.acl-main.747.pdf)                                | 278M           | 250K                | 1.5M steps / BS 8192  | 2.5T CC100 data         | 100                       |
| [XLM-R large](https://huggingface.co/xlm-roberta-large)                 | [Conneau et al. (2020)](https://aclanthology.org/2020.acl-main.747.pdf)                                | 560M           | 250K                | 1.5M steps / BS 8192  | 2.5T CC100 data         | 100                       |

In the following, we will provide the results on the basis of the multilingual models.

The final LEXTREME score is computed using the harmonic mean of the dataset and the language aggregate score.

### Dataset aggregate scores for multilingual models. The best scores are in bold.

We compute the dataset aggregate score by taking the successive harmonic mean of (1.) the languages inside the configurations (e.g., de,fr,it within SJP), (2.) the configurations inside the datasets (e.g., OTS-UL, OTS-CT within OTS), and (3.) the datasets inside LEXTREME (BCD, GAM, etc.).

| **Model**   | **BCD**  | **GAM**  | **GLC**  | **SJP**  | **OTS**  | **C19**  | **MEU**  | **GLN**  | **LNR**  | **LNB**  | **MAP**  | **Agg.** |
| ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| MiniLM      | 53.0     | **73.3** | 42.1     | 67.7     | 44.1     | 2.6      | 62.0     | 40.5     | 46.8     | 86.0     | 55.5     | 52.2     |
| DistilBERT  | 54.5     | 69.5     | **62.8** | 66.8     | 56.1     | 22.2     | 63.6     | 38.1     | 48.4     | 78.7     | 55.0     | 56.0     |
| mDeBERTa v3 | 57.6     | 70.9     | 52.2     | 69.1     | 66.5     | 25.5     | 65.1     | 42.2     | 46.6     | 87.8     | **60.2** | 58.5     |
| XLM-R base  | **63.5** | 72.0     | 56.8     | **69.3** | 67.8     | 26.4     | 65.6     | 47.0     | 47.7     | 86.0     | 56.1     | 59.9     |
| XLM-R large | 58.7     | 73.1     | 57.4     | 69.0     | **75.0** | **29.0** | **68.1** | **48.0** | **49.5** | **88.2** | 58.5     | **61.3** |

### Language aggregate scores for multilingual models. The best scores are in bold.

We compute the language aggregate score by taking the successive harmonic mean of (1.) the configurations inside the datasets, (2.) the datasets for the given language (e.g., MAP and MEU for lv), and (3.) the languages inside LEXTREME (bg,cs, etc.).

| **Model**   | **bg**   | **cs**   | **da**   | **de**   | **el**   | **en**   | **es**   | **et**   | **fi**   | **fr**   | **ga**   | **hr**   | **hu**   | **it**   | **lt**   | **lv**   | **mt**   | **nl**   | **pl**   | **pt**   | **ro**   | **sk**   | **sl**   | **sv**   | **Agg.** |
| ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| MiniLM      | 64.0     | 57.7     | 55.4     | 60.1     | 48.9     | 42.8     | 63.8     | 59.7     | 56.6     | 48.5     | 41.5     | 62.2     | 41.8     | 45.6     | 59.8     | 60.2     | 55.7     | 38.8     | 33.5     | 63.5     | 58.4     | 58.9     | 62.2     | 59.4     | 54.1     |
| DistilBERT  | 65.3     | 60.2     | 57.4     | 64.1     | 53.1     | 54.0     | 66.9     | 57.4     | 55.7     | 55.8     | 45.5     | 63.1     | 39.9     | 54.9     | 58.0     | 57.7     | 57.3     | 42.0     | 43.6     | 64.7     | 57.4     | 59.0     | 63.3     | 59.2     | 56.5     |
| mDeBERTa v3 | 61.9     | 60.6     | 59.3     | 66.6     | 54.0     | 58.9     | 66.9     | 60.3     | **61.1** | 57.0     | 50.2     | 65.0     | 44.2     | 59.7     | 63.7     | 61.4     | **61.2** | **48.1** | 50.2     | 67.9     | **60.8** | **65.2** | 65.2     | **65.4** | 59.8     |
| XLM-R base  | **68.3** | 61.3     | 58.5     | 66.0     | 54.7     | 58.6     | 63.8     | 59.3     | 57.5     | 57.7     | 47.8     | 65.9     | 43.3     | 59.6     | 60.3     | 60.8     | 58.0     | 45.0     | 52.0     | **68.2** | 59.2     | 60.3     | 66.2     | 61.7     | 58.9     |
| XLM-R large | 64.5     | **63.3** | **65.1** | **68.3** | **59.6** | **61.9** | **70.0** | **61.3** | 60.9     | **57.9** | **50.3** | **68.3** | **44.7** | **62.9** | **66.1** | **65.5** | 60.1     | 43.9     | **55.0** | 68.1     | 60.2     | 62.8     | **68.2** | 62.5     | **61.3** |

## Frequently Asked Questions (FAQ)

### Where are the datasets?

We provide access to LEXTREME at https://huggingface.co/datasets/joelito/lextreme.

For example, to load the swiss_judgment_prediction ([Niklaus et al. 2021](https://aclanthology.org/2021.nllp-1.3/)) dataset, you first simply install the datasets' python library and then make the following call:

```python

from datasets import load_dataset

dataset = load_dataset("joelito/lextreme", "swiss_judgment_prediction")

```

### How to run experiments?

It is possible to reproduce the results of the paper by running the finetuning for each dataset separately. Alternatively, you can run [main.py](https://github.com/JoelNiklaus/LEXTREME/tree/main/main.py) which, in a nutshell, will generate bash scripts for each dataset with the necessary hyperparameters and run them on every available GPU in your system (if available).

The following command will make sure that you run most experiments as described in the paper:

```
python main.py
```

It allows a certain degree of customizability by specifying the following arguments:

| short argument name | full argument name   | description                                                                                                                                                                                                                         | default value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| :------------------ | :------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| -as                 | --accumulation_steps | Define the number of accumulation_steps.                                                                                                                                                                                            | Generated automatically depending on the batch size and the size of the pretrained model                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -bz                 | --batch_size         | Define the batch size.                                                                                                                                                                                                              | Generated automatically depending on the size of the pretrained model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -gn                 | --gpu_number         | Define which GPU you would like to use. If you want to specify multiple GPUs, seperate the integers by a comma.                                                                                                                     | Available GPUs are detected automatically. If no GPU is available, the CPU is used instead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -gm                 | --gpu_memory         | Define how much memory your GPUs have. Depending on that the batch size will be calculated automatically. In case the batch size is too big, you can change it manually with `-bz`.                                                 | 11                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -hier               | --hierarchical       | Define whether you want to use a hierarchical model or not. Caution: this will not work for every task.                                                                                                                             | Defined automatically depending on the dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -lang               | --language           | Define if you want to filter the training dataset by language.                                                                                                                                                                      | `all`; only important for multlingual datasets; per default the entire dataset is used                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -lc                 | --lower_case         | Define if lower case or not.                                                                                                                                                                                                        | False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -lmt                | 50                   | Define which kind of language model you would like to use; you can choose between small,base and large language models or all of them.                                                                                              | `all` = all pretrained language models as decribed in the paper                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -los                | --list_of_seeds      | Define the number of training epochs.                                                                                                                                                                                               | Three seeds (1,2,3) are used                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -lr                 | --learning_rate      | Define the learning rate.                                                                                                                                                                                                           | 1e-05                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -nte                | --num_train_epochs   | Define the number of training epochs.                                                                                                                                                                                               | 50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -rmo                | --running_mode       | Define whether you want to run the finetungin on all available training data or just a small portion for testing purposes.                                                                                                          | `default` = the entire dataset will be used for finetuning. The other option is `experimental` which will only take a small fraction of the dataset for experimental purposes.                                                                                                                                                                                                                                                                                                                                                                                      |
| -t                  | --task               | Choose a specific task or all of them.                                                                                                                                                                                              | `all`. The other options are: `brazilian_court_decisions_judgment`, `brazilian_court_decisions_unanimity`, `covid19_emergency_event`, `german_argument_mining`, `greek_legal_code_chapter_level`, `greek_legal_code_subject_level`, `greek_legal_code_volume_level`, `greek_legal_ner`, `legalnero`, `lener_br`, `mapa_ner_coarse_grained`, `mapa_ner_fine_grained`, `multi_eurlex_level_1`, `multi_eurlex_level_2`, `multi_eurlex_level_3`, `online_terms_of_service_unfairness_category`, `online_terms_of_service_unfairness_level`, `swiss_judgment_prediction` |
| -dmo                | --download_mode      | Choose if you want to redownload the datasets or use load them from cache.                                                                                                                                                          | `force_redownload`. The other options are `reuse_dataset_if_exists`, `reuse_cache_if_exists`. For more information, visit: https://huggingface.co/docs/datasets/v1.4.1/loading_datasets.html.                                                                                                                                                                                                                                                                                                                                                                       |
| -od                 | --output_dir         | Specify the output directory for the logs.                                                                                                                                                                                          | Generated automatically with a time stamp.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -rev                | -revision            | The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git. | `main                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

For example, if you want to finetune on `swiss_judgment_prediction` with the seeds [1,2,3], 10 epochs, and all pretrained language models as described in the paper, you can type the following:

```
python main.py --task swiss_judgment_prediction -list_of_seeds
 1,2,3 --num_train_epochs 10
```

Temporary bash files will be created and saved in the folder [temporary_scripts](https://github.com/JoelNiklaus/LEXTREME/tree/main/temporary_scripts) and they will be run immediately. These bash files will be overwritten the next time you run `main.py`.

If you want to finetune only on, let's say, `xlm-roberta-large`, you can type the following command.

```
python main.py --task swiss_judgment_prediction -list_of_seeds
 1,2,3 --num_train_epochs 10 --language_model_type xlm-roberta-large
```

If, additionally, you don't want to make use of a hierarchical model (`swiss_judgment_prediction` makes use of hierarchical models due to the length of the input documents), you type the following.

```
python main.py --task swiss_judgment_prediction -list_of_seeds
 1,2,3 --num_train_epochs 10 --language_model_type xlm-roberta-large --hierarchical False
```

Not all tasks support the use of hierarchical types. For example, the code for the named entity recognition tasks has not been optimized to make use of both the non-hierarchical and the hierarchical variants. Therefore, setting `-hierarchical` to True will cause an error.

### How can I contribute a dataset to LEXTREME?

If you want to extend the benchmark with your own datasets, you can do so by following the following instructions:

#### _Make your dataset available on hugginface_

1. Make sure your dataset is available on the huggingface hub and has a train, validation and test split.
2. Make sure that the structure of your dataset is in compliance with the other datasets of LEXTREME.
3. Create a pull request to the lextreme repository by adding the following to the LEXTREME.py file:
   - Create a dict \_{YOUR_DATASET_NAME} (similar to \_BRAZILIAN_COURT_DECISIONS_JUDGMENT) containing all the necessary information about your dataset (task_type, input_col, label_col, etc.)
   - Add your dataset to the BUILDER\*CONFIGS list: `LextremeConfig(name="{your_dataset_name}", \*\**{YOUR_DATASET_NAME})`
   - Test that it works correctly by loading your subset with `load_dataset("lextreme", "{your_dataset_name}")` and inspecting a few examples.

#### _GitHub_

The following instructions will suffice only if

- your dataset is in compliance with the other datasets of LEXTREME and
- the tasks of your dataset belong to these classes: `token classification`, `single-label text classification`, `multi-label text classification`.

1. Navigate to the folder `utils` and open the file `meta_infos.json`.
2. The file contains several fields with important information about each dataset and finetuning task. Some of this information is essential to run the code. The fields are:

- `dataset_jurisdiction`: Not important for the code. Nevertheless, important to assess the jurisdictional coverage of LEXTREME.
- `dataset_abbreviations`: Not important for the code. Nevertheless, important to add the results of the finetuning to the existing tables.
- `task_abbreviations`: Not important for the code. Nevertheless, important to add the results of the finetuning to the existing tables.
- `task_type_mapping`: Important for the code. Specify to which type of task your dataset, e.g. the respective finetuning task, belongs to. Choose one of the following abbreviations:
  - NER (token classification/ named entity recognition)
  - SLTC (single-label text classification)
  - MLTC (multi-label text classification)
- `task_language_mapping`: Important for the code. Provide a list of languages that your finetuning task covers. Use only two-letter lowercase abbreviation. You can find an overview [here](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
- `config_to_dataset`: Not important for the code. Nevertheless, this information is useful.
- `dataset_to_config`: Not important for the code. Nevertheless, this information is useful.
- `task_default_arguments`: Important for the code. Here, you can define the arguments that will be served for finetuning. Have a look at the existing examples. Essentially, what you need to provide is `max_seq_length` and `hierarchical`. `max_segments` and `max_seg_length` are only needed if `hierarchical` is set to `true`.
- `language_models`: Important for the code. If your dataset covers a new language, you might want to add a new monolingual language model for that language. Provide the name as given on huggingface.

Once these steps are finished, make a merge request, and we merge the changes into the main branch.

#### Code

## References

Please cite the following preprint:

```
@misc{niklaus2023lextreme,
    title={LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain},
    author={Joel Niklaus and Veton Matoshi and Pooja Rani and Andrea Galassi and Matthias Stürmer and Ilias Chalkidis},
    year={2023},
    eprint={2301.13126},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
