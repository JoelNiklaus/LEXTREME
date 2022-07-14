import os
import argparse
from pathlib import Path
import re


os.chdir('./scripts')


models_to_be_used_small = open('models_to_be_used_small.txt','r').readlines()
models_to_be_used_small = [x.strip() for x in models_to_be_used_small]
models_to_be_used_base = open('models_to_be_used_base.txt','r').readlines()
models_to_be_used_base = [x.strip() for x in models_to_be_used_base]
models_to_be_used_large = open('models_to_be_used_large.txt','r').readlines()
models_to_be_used_large = [x.strip() for x in models_to_be_used_large]


'''
multi_linguagel_tasks = ["covid19_emergency_event","online_terms_of_service_unfairness_level","swiss_judgment_prediction","online_terms_of_service_unfairness_category","multi_eurlex"]


def adjust_language_list(task:str,language:str):

    pass
'''

def run_experiment(language_model_type='all',running_mode='default', task='all'):
    
    if language_model_type=='all':
        with open('models_to_be_used.txt','w') as f:
            for lm in models_to_be_used_small + models_to_be_used_base + models_to_be_used_large:
                print(lm,file=f)

    elif language_model_type=='small':
        with open('models_to_be_used.txt','w') as f:
            for lm in models_to_be_used_small:
                print(lm,file=f)

    elif language_model_type=='base':
        with open('models_to_be_used.txt','w') as f:
            for lm in models_to_be_used_base:
                print(lm,file=f)

    elif language_model_type=='large':
        with open('models_to_be_used.txt','w') as f:
            for lm in models_to_be_used_large:
                print(lm,file=f)

    with open('running_mode.txt','w') as f:
        print(running_mode,file=f)

    path = Path('./')

    scripts = [f for f in path.glob('*.sh') if f.is_file()]
    for script in scripts:
        if str(script).endswith('sh'):
            if task=='all':
                print(script)
                os.system('bash '+str(script))
            else:
                if bool(re.search(str(task),str(script)))==True:
                    print(script)
                    os.system('bash '+str(script))




if __name__=='__main__':

    list_of_tasks = os.listdir('./')
    list_of_tasks = [x.strip()[:-3] for x in list_of_tasks if x.endswith('sh')]
    list_of_tasks = [re.sub(r'run_','',x) for x in list_of_tasks]

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lmt','--language_model_type', help='Define which kind of language model you would like to use; you can choose between small, base and large language models or all of them.', default='all')
    parser.add_argument('-rmo','--running_mode', help='Define whether you want to run the finetungin on all available training data or just a small portion for testing purposes.', default='default')
    parser.add_argument('-t','--task', help='Choose a task.', default='all',
    choices=list_of_tasks
    )
    
    args = parser.parse_args()

    run_experiment(args.language_model_type,args.running_mode,args.task)

