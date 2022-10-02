# LEXTREME: A Multlingual Benchmark Dataset for Legal Language Understanding 

## Dataset Summary
[comming soon]
## Supported Tasks
[comming soon]
## LEXTREME Scores
[comming soon]
## Frequently Asked Questions (FAQ)

### Where are the datasets?
We provide access to LEXTREME at https://huggingface.co/datasets/joelito/lextreme.  

For example, to load the swiss_judgment_prediction ([Niklaus, Chalkidis, and Stürmer 2021](https://aclanthology.org/2021.nllp-1.3/)) dataset, you first simply install the datasets python library and then make the following call:

```python

from datasets import load_dataset

dataset = load_dataset("joelito/lextreme", "swiss_judgment_prediction")

```

### How to run experiments?

The folder [experiments](https://github.com/JoelNiklaus/LEXTREME/tree/main/experiments) contains all python scripts to run the finetuning for each task seperately. In order to do so, you need to provide at least a two arguments: ```output_dir```, i.e. where you want to store the output of the finetuning, and ```model_name_or_path```, i.e. the pretrained language model that you want to use (e.g. ```distilbert-base-multilingual-cased```). Optionally and depending on your hardware, you can specify the GPU number in order to speed up the process by providing ```CUDA_VISIBLE_DEVICES={GPU_NUMBER}```.

For example, if you want to finetune on the swiss_judgment_prediction``` dataset, you type the following command and replace the curly brackets and the content therein with your variables:  

```
CUDA_VISIBLE_DEVICES={GPU_NUMBER} python ```run_swiss_judgment_prediction.py --output_dir {OUTPUT_DIR} --model_name_or_path {MODEL_NAME_OR_PATH}

```

Note that the command will make use of the predefined configurations as described in the paper, i.e. it will train for 50 epochs with an early stopping patience of 5. In order to override these configurations, you can either change the code itself, or you can provide required arguments in the command line. For example, in order to reduce the number of epochs to, let's say, 3, you can do the following:

```
CUDA_VISIBLE_DEVICES={GPU_NUMBER} python run_swiss_judgment_prediction.py --output_dir {OUTPUT_DIR} --model_name_or_path {MODEL_NAME_OR_PATH} --num_train_epochs 3

```

### How reproduce the results of the paper?

It is possible to reproduce the results of the paper by running the finetung for each dataset separately. Alternatively, you can run ```main.py``` which, in a nutshell, will generate bash scripts for each dataset with the necessary hyperparameters and run them on every available GPU in your system (if available). 

The following command will make sure that you run all experiments as described in the paper:

```
python main.py
```

It allows a certain degree of customizability by specifying the following arguments:


| short argument name   | full argument name   | description                                                                                                                            | default value                                                                                                                                                                |
|:----------------------|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -as                   | --accumulation_steps | Define the number of accumulation_steps.                                                                                               | generated automatically depending on the batch size and the size of the pretrained model                                                                                     |
| -bz                   | --batch_size         | Define the batch size.                                                                                                                 | generated automatically depending on the size of the pretrained model                                                                                                        |
| -gn                   | --gpu_number         | Define which gpu you would like to use.                                                                                                | detected automatically                                                                                                                                                       |
| -hier                 | --hierarchical       | Define whether you want to use a hierarchical model or not. Caution: this will not work for every task.                                | defined automatically depending on the dataset                                                                                                                               |
| -lang                 | --language           | Define if you want to filter the training dataset by language.                                                                         | all_languages; only important for multlingual datasets; per default the entire dataset is used                                                                               |
| -lc                   | --lower_case         | Define if lower case or not.                                                                                                           | False                                                                                                                                                                        |
| -lmt                  | 50                   | Define which kind of language model you would like to use; you can choose between small,base and large language models or all of them. | all = all pretrained language models as decribed in the paper                                                                                                                |
| -los                  | --list_of_seeds      | Define the number of training epochs.                                                                                                  | None = five seeds (1,2,3,4,5) are used                                                                                                                                       |
| -lr                   | --learning_rate      | Define the learning rate.                                                                                                              | 1e-05                                                                                                                                                                        |
| -nte                  | --num_train_epochs   | Define the number of training epochs.                                                                                                  | 50                                                                                                                                                                           |
| -rmo                  | --running_mode       | Define whether you want to run the finetungin on all available training data or just a small portion for testing purposes.             | default = the entire dataset will be used for finetuning. The other option is "experimental" which will only take a small fraction of the dataset for experimental purposes. |
| -t                    | --task               | Choose a specific task or all of them.                                                                                                 | all                                                                                                                                                                          |


For example, if you want to finetune on swiss_judgment_prediction with the seeds [1,2,3], 10 epochs, and all pretrained language models as described in the paper, you can type the following:

```
python main.py --task swiss_judgment_prediction -python main.py --task swiss_judgment_prediction -list
 1,2,3 --num_train_epochs 10
```

Temporary bash files will be created and saved in the folder ```temporary_scripts``` and they will be run immediately. These bash files will be overwritten the next time you run main.py.

If you want to finetune only on, let's say, ```xlm-roberta-large```.
```
python main.py --task swiss_judgment_prediction -python main.py --task swiss_judgment_prediction -list
 1,2,3 --num_train_epochs 10 --language_model_type xlm-roberta-large
```

If, additionally, you don't want to make use of a hierarchical model (```swiss_judgment_prediction``` makes use of hierarchical models due to the length of the input documents), you type the following.
```
python main.py --task swiss_judgment_prediction -python main.py --task swiss_judgment_prediction -list
 1,2,3 --num_train_epochs 10 --language_model_type xlm-roberta-large --hierarchical False
```


## References

```
@inproceedings{niklaus-etal-2021-swiss,
    title = "{S}wiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark",
    author = {Niklaus, Joel  and
      Chalkidis, Ilias  and
      St{\"u}rmer, Matthias},
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.3",
    doi = "10.18653/v1/2021.nllp-1.3",
    pages = "19--35",
    abstra
´´´