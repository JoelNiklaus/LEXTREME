from datasets import Dataset
import pandas as pd
from typing import List
import os
from pathlib import Path
import json as js
from datasets import Dataset, load_metric
from transformers import EvalPrediction
from scipy.special import expit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
import numpy as np
import datetime
import wandb
import re
from transformers import (
    AutoConfig,
    BertConfig,
    RobertaConfig,
    DebertaConfig,
    AutoTokenizer,
    BertTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
    DebertaForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,

)


def split_into_languages(dataset):
    dataset_new = list()
    
    dataset_df = pd.DataFrame(dataset)
    
    for item in dataset_df.to_dict(orient='records'):
        celex_id = item['celex_id']
        labels = item['labels']
        for k,v in item.items():
            if k=='text':
                for language, document in v.items():
                    item_new = dict()
                    item_new['language']=language
                    item_new['text']=document
                    item_new['celex_id']=celex_id
                    item_new['labels']=labels
                    dataset_new.append(item_new)
    
    dataset_new = pd.DataFrame(dataset_new)
    
    dataset_new = Dataset.from_pandas(dataset_new)
    
    return  dataset_new


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
    accuracy_not_noramlized = accuracy_score(y_true=p.label_ids, y_pred=preds, normalize=False)
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
            #'mcc':mcc, 
            'accuracy_normalized':accuracy_normalized,
            'accuracy_not_noramlized':accuracy_not_noramlized,
            'macro-precision': precision_macro,
            'micro-precision':precision_micro,
            'weighted-precision': precision_weighted,
            'macro-recall_score':recall_score_macro,
            'micro-recall_score': recall_score_micro,
            'weighted-recall_score': recall_score_weighted
            }



# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics_multi_class(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='weighted', zero_division=0)
    accuracy_not_noramlized = accuracy_score(y_true=p.label_ids, y_pred=preds, normalize=False)
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
            'accuracy_not_noramlized':accuracy_not_noramlized,
            'macro-precision': precision_macro,
            'micro-precision':precision_micro,
            'weighted-precision': precision_weighted,
            'macro-recall_score':recall_score_macro,
            'micro-recall_score': recall_score_micro,
            'weighted-recall_score': recall_score_weighted
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
            ['I-'+self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            ['I-'+self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        for k in results.keys():
            if(k not in flattened_results.keys()):
                flattened_results[k+"_f1"]=results[k]["f1"]


        return flattened_results


def convert_id2label(label_output:list,id2label:dict)->list:

    label_output = list(label_output)
    
    label_output_indices_of_positive_values = list()
    
    for n, lp in enumerate(label_output):
        if lp==1:
            label_output_indices_of_positive_values.append(n)

    label_output_as_labels = [id2label[l] for l in label_output_indices_of_positive_values]

    return label_output_as_labels







def make_predictions_multi_class(trainer,data_args,predict_dataset,id2label,training_args,list_of_languages=[],name_of_input_field='input'):
    
    language_specific_metrics = list()
    if data_args.language=='all_languages':
        
        for l in list_of_languages:
            
            predict_dataset_filtered = predict_dataset.filter(lambda example: example['language']==l)

            predictions, labels, metrics = trainer.predict(predict_dataset_filtered, metric_key_prefix=l+"_predict")

            language_specific_metrics.append(metrics)
        
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    

    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    language_specific_metrics.append(metrics)

    language_specific_metrics = merge_dicts(language_specific_metrics)

    trainer.log_metrics("predict", language_specific_metrics)
    trainer.save_metrics("predict", language_specific_metrics)


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



def make_predictions_multi_label(trainer,data_args,predict_dataset,id2label,training_args,list_of_languages=[],name_of_input_field='input'):
    
    language_specific_metrics = list()
    
    if "language" in list(predict_dataset.features.keys()):
        if data_args.language=='all_languages':
            
            for l in list_of_languages:
                
                predict_dataset_filtered = predict_dataset.filter(lambda example: example['language']==l)

                if len(predict_dataset_filtered['language'])>0:

                    predictions, labels, metrics = trainer.predict(predict_dataset_filtered, metric_key_prefix=l+"_predict")

                    language_specific_metrics.append(metrics)
        

    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    

    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    language_specific_metrics.append(metrics)

    language_specific_metrics = merge_dicts(language_specific_metrics)

    trainer.log_metrics("predict", language_specific_metrics)
    trainer.save_metrics("predict", language_specific_metrics)


    output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            for index, pred_list in enumerate(predictions):
                pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                writer.write(f"{index}\t{pred_line}\n")

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
        
        for l in list_of_languages:
            
            predict_dataset_filtered = predict_dataset.filter(lambda example: example['language']==l)

            if len(predict_dataset_filtered['language'])>0:

                predictions, labels, metrics = trainer.predict(predict_dataset_filtered, metric_key_prefix=l+"_predict")

                language_specific_metrics.append(metrics)


        

    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    language_specific_metrics.append(metrics)

    language_specific_metrics = merge_dicts(language_specific_metrics)

    trainer.log_metrics("predict", language_specific_metrics)
    trainer.save_metrics("predict", language_specific_metrics)

    
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



def config_wandb(training_args, model_args, data_args):

    time_now = datetime.datetime.now().isoformat()
    time_now = datetime.datetime.now().isoformat()
    project_name = "LEXTREME-"+data_args.finetuning_task
    wandb.init(project=project_name)
    wandb.run.name=data_args.finetuning_task+'_'+model_args.model_name_or_path+'_seed-'+str(training_args.seed)+'__time-'+time_now

def get_optimal_max_length(tokenizer, train_dataset, eval_dataset, predict_dataset):
    all_inputs = train_dataset['input'] + eval_dataset['input'] + predict_dataset['input']
    all_inputs = [len(tokenizer(i)['input_ids']) for i in all_inputs]
    max_length = max(all_inputs)
    if max_length <=512:
        return max_length
    else:
        return 512



def generate_Model_Tokenizer_for_SequenceClassification(model_args, data_args, num_labels):

    model_types = {
                    "MiniLM": ['microsoft/Multilingual-MiniLM-L12-H384'],
                    "bert": ["distilbert-base-multilingual-cased"],
                    "deberta" : ["microsoft/mdeberta-v3-base"],
                    "roberta" : ["xlm-roberta-base","xlm-roberta-large"]
    }

    
    if model_args.model_name_or_path in model_types['bert']:
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = BertConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if config.model_type == 'big_bird':
            config.attention_type = 'original_full'

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif model_args.model_name_or_path in model_types['roberta']:
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = RobertaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task= data_args.language+'_'+data_args.finetuning_task,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if config.model_type == 'big_bird':
            config.attention_type = 'original_full'

        tokenizer = RobertaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = RobertaForSequenceClassification.from_pretrained(
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

        if config.model_type == 'big_bird':
            config.attention_type = 'original_full'

        # DebertaTokenizer is buggy, therefore we use the AutTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
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

        if config.model_type == 'big_bird':
            config.attention_type = 'original_full'

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
    

    return model, tokenizer