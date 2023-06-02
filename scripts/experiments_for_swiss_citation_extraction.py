import json as js
import os
from pprint import pprint
with open('../utils/report_specs/swiss_citation_extraction.json', 'r') as f:
    report_specs = js.load(f)
    pprint(report_specs)

required_models = [
    "microsoft/Multilingual-MiniLM-L12-H384",
    "distilbert-base-multilingual-cased",
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "ZurichNLP/swissbert",
    "ZurichNLP/swissbert-xlm-vocab",
    "joelito/legal-swiss-roberta-base",
    "joelito/legal-swiss-roberta-large",
    "joelito/legal-swiss-longformer-base"
  ]

os.chdir('../')

for model_and_revision in report_specs["_name_or_path"]:
    name_or_path = model_and_revision.split('@')[0]
    if name_or_path in required_models:
        revision = model_and_revision.split('@')[1]
        ft = 'original'
        seeds = '1,2,3'
        command = 'python main.py -gn 0,1 -gm 80 -los ' + seeds + \
                  ' -ld swiss_citation_extraction ' + \
                  ' -t ' + ft + ' -lmt ' + name_or_path + ' -rev ' + revision
        os.system(command)
