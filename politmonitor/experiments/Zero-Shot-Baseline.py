#!/usr/bin/env python
# coding: utf-8

import sys
from transformers import pipeline
from sklearn.metrics import f1_score
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm


sys.path.append('../code/')

from utils import LabelHandler
from training_data_handler import TrainingDataHandler


tqdm.pandas()

tdh = TrainingDataHandler()
lh = LabelHandler()


device = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'

print('Device: ', device)

classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device=device)


language = 'all'
tdh.get_training_data(language=language, affair_text_scope=[
                      'zh', 'ch'], affair_attachment_category='all')
test_dataset = tdh.training_data_df[tdh.training_data_df.split == 'test']


# Inserting columns for candidate labels
test_dataset['candidate_labels_zero_shot'] = test_dataset.language.apply(
    lh.get_all_labels_per_language)


def classify(sequence_to_classify, label_list=None):

    if label_list is None:
        label_list = list(tdh.label2id.keys())

    output = classifier(sequence_to_classify, label_list, multi_label=True)
    labels_scores = list(zip(output["labels"], output["scores"]))
    labels_scores = [res[0] for res in labels_scores if res[1] > 0.5]
    return labels_scores


test_dataset['zero_shot_results'] = test_dataset.progress_apply(
    lambda x: classify(x['text'], x['candidate_labels_zero_shot']), axis=1)


test_dataset.to_json('results_zero_shot.jsonl', lines=True,
                     orient='records', force_ascii=False)
