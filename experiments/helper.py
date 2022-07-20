from datasets import Dataset
import pandas as pd
from typing import List
import os
from pathlib import Path
import json as js
from datasets import Dataset, load_metric
from transformers import EvalPrediction
from scipy.special import expit
from sklearn.metrics import f1_score
import numpy as np
from typing import Tuple


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



def reduce_size(dataset,n):
    dataset_df = pd.DataFrame(dataset)
    dataset_df = dataset_df[:n]
    dataset_new = Dataset.from_pandas(dataset_df)
    return dataset_new

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics_multi_label(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)

    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}



# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics_multi_class(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}






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

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
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












