from datasets import Dataset
import pandas as pd
from typing import List
import os
from pathlib import Path
import json as js
from ast import literal_eval
from datasets import Dataset, load_metric, concatenate_datasets
from transformers import EvalPrediction
from scipy.special import expit
from sklearn.utils.extmath import softmax
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from seqeval.metrics import accuracy_score as seqeval_accuracy_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from seqeval.scheme import IOB2
import numpy as np
import datetime
import wandb
import re
import sys


from transformers import (
    AutoConfig,
    DistilBertConfig,
    XLMRobertaConfig,
    DebertaConfig,
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    DebertaForSequenceClassification,
    BertForSequenceClassification,
    DistilBertForTokenClassification,
    XLMRobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    XLMRobertaForTokenClassification,
    DebertaForTokenClassification

)


from models.deberta import HierDebertaForSequenceClassification
from models.distilbert import HierDistilBertForSequenceClassification


def append_zero_segments(case_encodings, pad_token_id, data_args):
        """appends a list of zero segments to the encodings to make up for missing segments"""
        return case_encodings + [[pad_token_id] * data_args.max_seg_length] * (
                data_args.max_segments - len(case_encodings))


def add_oversampling_to_multiclass_dataset(train_dataset,id2label,data_args):

    # NOTE: This is not optimized for multiclass classification
    
    label_datasets = dict()
    minority_len, majority_len = len(train_dataset), 0
    for label_id in id2label.keys():
        label_datasets[label_id] = train_dataset.filter(lambda item: item['label'] == label_id,
                                                        load_from_cache_file=data_args.overwrite_cache)
        if len(label_datasets[label_id]) < minority_len:
            minority_len = len(label_datasets[label_id])
            minority_id = label_id
        if len(label_datasets[label_id]) > majority_len:
            majority_len = len(label_datasets[label_id])
            majority_id = label_id

    datasets = [train_dataset]
    num_full_minority_sets = int(majority_len / minority_len)
    for i in range(num_full_minority_sets - 1):  # -1 because one is already included in the training dataset
        datasets.append(label_datasets[minority_id])

    remaining_minority_samples = majority_len % minority_len
    random_ids = np.random.choice(minority_len, remaining_minority_samples, replace=False)
    datasets.append(label_datasets[minority_id].select(random_ids))
    train_dataset = concatenate_datasets(datasets)

    return train_dataset

def preprocess_function(batch, tokenizer, model_args, data_args, id2label=None):


    '''Can be used with any task that requires hierarchical models'''

    if type(id2label)==dict:
        batch["labels"] = [[1 if label in labels else 0 for label in list(id2label.keys())] for labels in batch["label"]]
        del batch["label"]


    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    pad_id = tokenizer.pad_token_id


    if model_args.hierarchical == True:
        batch['segments'] = []

        tokenized = tokenizer(batch["input"], padding=padding, truncation=True,
                                max_length=data_args.max_segments * data_args.max_seg_length,
                                add_special_tokens=False)  # prevent it from adding the cls and sep tokens twice
        
        for ids in tokenized['input_ids']:
            # convert ids to tokens and then back to strings
            id_blocks = [ids[i:i + data_args.max_seg_length] for i in range(0, len(ids), data_args.max_seg_length) if
                            ids[i] != pad_id]  # remove blocks containing only ids
            id_blocks[-1] = [id for id in id_blocks[-1] if
                                id != pad_id]  # remove remaining pad_tokens_ids from the last block
            token_blocks = [tokenizer.convert_ids_to_tokens(ids) for ids in id_blocks]
            string_blocks = [tokenizer.convert_tokens_to_string(tokens) for tokens in token_blocks]
            batch['segments'].append(string_blocks)
            
        
        # Tokenize the text
        
        tokenized = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        for case in batch['segments']:
            case_encodings = tokenizer(case[:data_args.max_segments], padding=padding, truncation=True,
                                    max_length=data_args.max_seg_length, return_token_type_ids=True)
            tokenized['input_ids'].append(append_zero_segments(case_encodings['input_ids'], pad_id, data_args))
            tokenized['attention_mask'].append(append_zero_segments(case_encodings['attention_mask'], 0, data_args))
            tokenized['token_type_ids'].append(append_zero_segments(case_encodings['token_type_ids'], 0, data_args))

        
        del batch['segments']

    elif model_args.hierarchical == False:
        tokenized = tokenizer(
            batch["input"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            )

    return tokenized

def split_into_languages(dataset):
    dataset_new = list()
    
    dataset_df = pd.DataFrame(dataset)
    
    for item in dataset_df.to_dict(orient='records'):
        labels = item['label']
        for language, document in literal_eval(item['input']).items():
            if document is not None:
                item_new = dict()
                item_new['language']=language
                item_new['input']=str(document)
                item_new['label']=labels
                dataset_new.append(item_new)
    
    dataset_new = pd.DataFrame(dataset_new)
    
    dataset_new = Dataset.from_pandas(dataset_new)
    
    return  dataset_new
    

def split_into_segments(text, max_seg_length):

    text_split = text.split()

    for i in range(0, len(text_split), max_seg_length):
        yield ' '.join(text_split[i:i + max_seg_length])



def get_label_dict(main_path:str,label_column:str='label')->dict:

    all_labels = set()
    label_dict = dict()
    for p in os.listdir(main_path):
        p = Path(main_path) / p
        df = pd.read_json(p, lines=True)

        for l in df[label_column].unique():
            all_labels.add(l)


    for n, l in enumerate(list(all_labels)):
        label_dict[l]=n

    return label_dict

def merge_dicts(dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def create_dataset(path:str,label_dict:dict,text_column:str='text', label_column:str='label'):
    
    '''
    This function reads the train, validation and test dataset
    from the Brazilian court decision task dataset from the jsonl format.
    We can use this script for the time being, since we haven't uploaded
    the task dataset to huggingface yet. 
    
    '''

    all_data = pd.read_json(path, lines=True)

    all_data['text']=all_data[text_column]
    all_data['label']=all_data[label_column]

    all_data = all_data[['text','label']]

    print(all_data.head())

    all_data['label']=all_data.label.apply(lambda x: label_dict[x])

    dataset = Dataset.from_pandas(all_data)

    return dataset



def save_metrics(split,metric_results:dict,output_path:str):

    if split in ['train','eval','predict']:
        with open(output_path+'/'+split+'_results.json','w') as f:
            js.dump(metric_results,f,ensure_ascii=False,indent=2)


# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics_multi_label(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    accuracy_not_normalized = accuracy_score(y_true=p.label_ids, y_pred=preds, normalize=False)
    accuracy_normalized = accuracy_score(y_true=p.label_ids, y_pred=preds, normalize=True)
    precision_macro = precision_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    precision_micro = precision_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    recall_score_macro = recall_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    recall_score_micro = recall_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    recall_score_weighted = recall_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    #mcc = matthews_corrcoef(y_true=p.label_ids, y_pred=preds)
    
    return {'macro-f1': macro_f1, 
            'micro-f1': micro_f1,
            'weighted-f1':weighted_f1,
            'accuracy_normalized':accuracy_normalized,
            'accuracy_not_normalized':accuracy_not_normalized,
            'macro-precision': precision_macro,
            'micro-precision':precision_micro,
            'weighted-precision': precision_weighted,
            'macro-recall':recall_score_macro,
            'micro-recall': recall_score_micro,
            'weighted-recall': recall_score_weighted
            }



# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics_multi_class(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    accuracy_not_normalized = accuracy_score(y_true=p.label_ids, y_pred=preds, normalize=False)
    accuracy_normalized = accuracy_score(y_true=p.label_ids, y_pred=preds, normalize=True)
    precision_macro = precision_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    precision_micro = precision_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    recall_score_macro = recall_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    recall_score_micro = recall_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    recall_score_weighted = recall_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true=p.label_ids, y_pred=preds)
    return {'macro-f1': macro_f1, 
            'micro-f1': micro_f1,
            'weighted-f1':weighted_f1,
            'mcc':mcc, 
            'accuracy_normalized':accuracy_normalized,
            'accuracy_not_normalized':accuracy_not_normalized,
            'macro-precision': precision_macro,
            'micro-precision':precision_micro,
            'weighted-precision': precision_weighted,
            'macro-recall':recall_score_macro,
            'micro-recall': recall_score_micro,
            'weighted-recall': recall_score_weighted
            }






# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.

class Seqeval():

    def __init__(self, label_list):
        self.metric = load_metric('seqeval')
        self.label_list = label_list

    

    def compute_metrics_for_token_classification(self,p: EvalPrediction):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        # Adding the prefix I- because the seqeval cuts the first letter because it, presumably, assumes that one of these tagging schemes has been used: ["IOB1", "IOB2", "IOE1", "IOE2", "IOBES", "BILOU"]
        # By adding the prefix I- we make sure that the labels returned are the original labels
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)

        macro_f1 = seqeval_f1_score(true_predictions, true_labels, average="macro", zero_division=0, mode='strict', scheme=IOB2)
        micro_f1 = seqeval_f1_score(true_predictions, true_labels, average="micro", zero_division=0, mode='strict', scheme=IOB2)
        weighted_f1 = seqeval_f1_score(true_predictions, true_labels, average="weighted", zero_division=0, mode='strict', scheme=IOB2)

        macro_precision = seqeval_precision_score(true_predictions, true_labels, average="macro", zero_division=0, mode='strict', scheme=IOB2)
        micro_precision = seqeval_precision_score(true_predictions, true_labels, average="micro", zero_division=0, mode='strict', scheme=IOB2)
        weighted_precision = seqeval_precision_score(true_predictions, true_labels, average="weighted", zero_division=0, mode='strict', scheme=IOB2)


        macro_recall = seqeval_recall_score(true_predictions, true_labels, average="macro", zero_division=0, mode='strict', scheme=IOB2)
        micro_recall = seqeval_recall_score(true_predictions, true_labels, average="micro", zero_division=0, mode='strict', scheme=IOB2)
        weighted_recall = seqeval_recall_score(true_predictions, true_labels, average="weighted", zero_division=0, mode='strict', scheme=IOB2)

        accuracy_normalized = seqeval_accuracy_score(true_predictions, true_labels) #mode and scheme cannot be used with accuracy
        
        flattened_results = {
            "macro-f1":macro_f1,
            "micro-f1":micro_f1,
            'weighted-f1':weighted_f1,
            "macro-precision":macro_precision,
            "micro-precision":micro_precision,
            "weighted-precision":weighted_precision,
            "macro-recall":macro_recall,
            "micro-recall":micro_recall,
            "weighted-recall":weighted_recall,
            'accuracy_normalized':accuracy_normalized
        }
        for k in results.keys():
            if k.startswith("overall")==False:
                flattened_results[k+"_f1"]=results[k]["f1"]
                flattened_results[k+"_precision"]=results[k]["precision"]
                flattened_results[k+"_recall"]=results[k]["recall"]


        return flattened_results


def convert_id2label(label_output:list,id2label:dict)->list:

    label_output = list(label_output)
    
    label_output_indices_of_positive_values = list()
    
    for n, lp in enumerate(label_output):
        if lp==1:
            label_output_indices_of_positive_values.append(n)

    label_output_as_labels = [id2label[l] for l in label_output_indices_of_positive_values]

    return label_output_as_labels



def process_results(preds, labels):
    preds = preds[0] if isinstance(preds, tuple) else preds
    probs = softmax(preds)
    preds = np.argmax(preds, axis=1)
    return preds, labels, probs



def make_predictions_multi_class(trainer,data_args,predict_dataset,id2label,training_args,list_of_languages=[],name_of_input_field='input'):
    
    language_specific_metrics = list()
    if data_args.language=='all_languages':

        if list_of_languages==[]:
            list_of_languages = sorted(list(set(predict_dataset['language'])))
        
        for l in list_of_languages:
            
            predict_dataset_filtered = predict_dataset.filter(lambda example: example['language']==l)

            if len(predict_dataset_filtered['language'])>0:
                metric_prefix = l+"_predict/"
                predictions, labels, metrics = trainer.predict(predict_dataset_filtered, metric_key_prefix=metric_prefix)
                wandb.log(metrics)

                language_specific_metrics.append(metrics)
        
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict/")

    predictions = predictions[0] if isinstance(predictions, tuple) else predictions

    

    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    language_specific_metrics.append(metrics)

    language_specific_metrics = merge_dicts(language_specific_metrics)

    trainer.log_metrics("predict", language_specific_metrics)
    trainer.save_metrics("predict", language_specific_metrics)
    wandb.log(language_specific_metrics)


    output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            for index, pred_list in enumerate(predictions):
                pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                writer.write(f"{index}\t{pred_line}\n")


    predict_dataset_df = pd.DataFrame(predict_dataset)

    preds = np.argmax(predictions, axis=-1)

    output = list(zip(predict_dataset_df[name_of_input_field].tolist(),labels,preds,predictions))
    output = pd.DataFrame(output, columns = ['text','reference','predictions','logits'])
    
    output['predictions_as_label']=output.predictions.apply(lambda x: id2label[x])
    output['reference_as_label']=output.reference.apply(lambda x: id2label[x])
    
    output_predict_file_new_json = os.path.join(training_args.output_dir, "test_predictions_clean.json")
    output_predict_file_new_csv = os.path.join(training_args.output_dir, "test_predictions_clean.csv")
    output.to_json(output_predict_file_new_json, orient='records', force_ascii=False)
    output.to_csv(output_predict_file_new_csv)

    return language_specific_metrics



def make_predictions_multi_label(trainer,data_args,predict_dataset,id2label,training_args,list_of_languages=[],name_of_input_field='input'):
    
    language_specific_metrics = list()
    
    if "language" in list(predict_dataset.features.keys()):
        if data_args.language=='all_languages':
            if list_of_languages==[]:
                list_of_languages = sorted(list(set(predict_dataset['language'])))
            
            for l in list_of_languages:
                
                predict_dataset_filtered = predict_dataset.filter(lambda example: example['language']==l)

                if len(predict_dataset_filtered['language'])>0:
                    metric_prefix = l+"_predict/"
                    predictions, labels, metrics = trainer.predict(predict_dataset_filtered, metric_key_prefix=metric_prefix)
                    wandb.log(metrics)

                    language_specific_metrics.append(metrics)
        

    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict/")

    predictions = predictions[0] if isinstance(predictions, tuple) else predictions


    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    language_specific_metrics.append(metrics)

    language_specific_metrics = merge_dicts(language_specific_metrics)

    trainer.log_metrics("predict", language_specific_metrics)
    trainer.save_metrics("predict", language_specific_metrics)
    wandb.log(language_specific_metrics)


    predict_dataset_df = pd.DataFrame(predict_dataset)

    preds = (expit(predictions) > 0.5).astype('int32')
    

    output = list(zip(predict_dataset_df[name_of_input_field].tolist(),labels,preds,predictions))
    output = pd.DataFrame(output, columns = ['sentence','reference','predictions','logits'])
    
    output['predictions_as_label']=output.predictions.apply(lambda x: convert_id2label(x,id2label))
    output['reference_as_label']=output.reference.apply(lambda x: convert_id2label(x,id2label))

    output_predict_file_new_json = os.path.join(training_args.output_dir, "test_predictions_clean.json")
    output_predict_file_new_csv = os.path.join(training_args.output_dir, "test_predictions_clean.csv")
    output.to_json(output_predict_file_new_json, orient='records', force_ascii=False)
    output.to_csv(output_predict_file_new_csv)



def make_predictions_ner(trainer,tokenizer,data_args,predict_dataset,id2label,training_args,list_of_languages=[]):
    
    language_specific_metrics = list()
    if data_args.language=='all_languages':
        if list_of_languages==[]:
            list_of_languages = sorted(list(set(predict_dataset['language'])))
        
        for l in list_of_languages:
            
            predict_dataset_filtered = predict_dataset.filter(lambda example: example['language']==l)

            if len(predict_dataset_filtered['language'])>0:
                metric_prefix = l+"_predict/"
                predictions, labels, metrics = trainer.predict(predict_dataset_filtered, metric_key_prefix=metric_prefix)
                wandb.log(metrics)

                language_specific_metrics.append(metrics)


        

    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict/")

    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    language_specific_metrics.append(metrics)

    language_specific_metrics = merge_dicts(language_specific_metrics)

    trainer.log_metrics("predict", language_specific_metrics)
    trainer.save_metrics("predict", language_specific_metrics)
    wandb.log(language_specific_metrics)

    
    preds = np.argmax(predictions, axis=2)
    textual_data = tokenizer.batch_decode(predict_dataset['input_ids'],skip_special_tokens=True)

    words = list()
    preds_final = list()
    references = list()

    for n, _ in enumerate(textual_data):
        w = textual_data[n].split()
        l = labels[n]
        p = preds[n]
        
        #Remove all labels with value -100
        valid_indices = [n for n,x in enumerate(list(l))if x!=-100]
        l = l[valid_indices]
        p = p[valid_indices]

        words.append(w)
        references.append(l)
        preds_final.append(p)
        
    output = list(zip(words,references,preds_final,predictions))
    output = pd.DataFrame(output, columns = ['words','references','predictions','logits'])
    
    output['predictions_as_label']=output.predictions.apply(lambda l: [id2label[x] for x in l])
    output['references_as_label']=output.references.apply(lambda l: [id2label[x] for x in l])

    output_predict_file_new_json = os.path.join(training_args.output_dir, "test_predictions_clean.json")
    output_predict_file_new_csv = os.path.join(training_args.output_dir, "test_predictions_clean.csv")
    output.to_json(output_predict_file_new_json, orient='records', force_ascii=False)
    output.to_csv(output_predict_file_new_csv)



def config_wandb(training_args, model_args, data_args, project_name=None):

    time_now = datetime.datetime.now().isoformat()
    time_now = datetime.datetime.now().isoformat()
    if project_name is None:
        project_name = 'bfh_test_hierachical'
        #project_name = model_args.model_name_or_path
        project_name = re.sub('/','-',project_name)
    wandb.init(project=project_name)
    try:
        run_name = data_args.finetuning_task+'_'+model_args.model_name_or_path+'_seed-'+str(training_args.seed)+'__hierarchical_'+str(model_args.hierarchical)+'__time-'+time_now
    except:
        run_name = data_args.finetuning_task+'_'+model_args.model_name_or_path+'_seed-'+str(training_args.seed)+'__time-'+time_now
    wandb.run.name=run_name

def get_optimal_max_length(tokenizer, train_dataset, eval_dataset, predict_dataset):
    all_inputs = train_dataset['input'] + eval_dataset['input'] + predict_dataset['input']
    all_inputs = [len(tokenizer(i)['input_ids']) if type(i)==str else len(i) for i in all_inputs]
    max_length = max(all_inputs)
    if max_length <=512:
        return max_length
    else:
        return 512



def generate_Model_Tokenizer_for_SequenceClassification(model_args, data_args, num_labels):

    model_types = {
                    "MiniLM": ['microsoft/Multilingual-MiniLM-L12-H384'],
                    "distilbert": ["distilbert-base-multilingual-cased"],
                    "deberta" : ["microsoft/mdeberta-v3-base"],
                    "xlm-roberta" : ["xlm-roberta-base","xlm-roberta-large"]
    }

    
    if model_args.model_name_or_path in model_types['distilbert']:
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = DistilBertConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = DistilBertTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if model_args.hierarchical==True:
            model = HierDistilBertForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                ignore_mismatched_sizes=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        

        if model_args.hierarchical==False:
            model = DistilBertForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                ignore_mismatched_sizes=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    
    elif model_args.model_name_or_path in model_types["xlm-roberta"]:
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # RobertaTokenizer yielded errors, therefore I used RobertaTokenizerFast
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            #use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if model_args.hierarchical==True:
            model = XLMRobertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                ignore_mismatched_sizes=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        if model_args.hierarchical==False:
            model = XLMRobertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                ignore_mismatched_sizes=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    elif model_args.model_name_or_path in model_types['deberta']:

        config = DebertaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # DebertaTokenizer is buggy, therefore we use the AutTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if model_args.hierarchical==True:
            model = HierDebertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                ignore_mismatched_sizes=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif model_args.hierarchical==False:
            model = DebertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                ignore_mismatched_sizes=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    
    elif model_args.model_name_or_path in model_types['MiniLM']:
        
        
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384: They explicitly state that "This checkpoint uses BertModel with XLMRobertaTokenizer so AutoTokenizer won't work with this checkpoint!".
        tokenizer = XLMRobertaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384: They state: Multilingual MiniLM uses the same tokenizer as XLM-R. But the Transformer architecture of our model is the same as BERT.
        model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    

    return model, tokenizer, config


def generate_Model_Tokenizer_for_TokenClassification(model_args, data_args, num_labels):

    model_types = {
                    "MiniLM": ['microsoft/Multilingual-MiniLM-L12-H384'],
                    "distilbert": ["distilbert-base-multilingual-cased"],
                    "deberta" : ["microsoft/mdeberta-v3-base"],
                    "roberta" : ["xlm-roberta-base","xlm-roberta-large"]
    }

    
    if model_args.model_name_or_path in model_types['distilbert']:
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = DistilBertConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            #use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        model = DistilBertForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif model_args.model_name_or_path in model_types["xlm-roberta"]:
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = XLMRobertaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # RobertaTokenizer yielded errors, therefore I used RobertaTokenizerFast
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            #use_fast=True,
            add_prefix_space=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = XLMRobertaForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif model_args.model_name_or_path in model_types['deberta']:

        config = DebertaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # DebertaTokenizer is buggy, therefore we use the AutTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            #use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = DebertaForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    elif model_args.model_name_or_path in model_types['MiniLM']:
        
        
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384: They explicitly state that "This checkpoint uses BertModel with XLMRobertaTokenizer so AutoTokenizer won't work with this checkpoint!".
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            #use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384: They state: Multilingual MiniLM uses the same tokenizer as XLM-R. But the Transformer architecture of our model is the same as BERT.
        model = BertForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    

    return model, tokenizer
