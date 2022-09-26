from collections import defaultdict
from itertools import cycle
from multiprocessing import Pool
import argparse
import datetime
import itertools
import json as js
import os
import setproctitle
import shutil
import torch



#Empty folder with temporary scripts
shutil.rmtree('./temporary_scripts', ignore_errors=True)
os.mkdir('./temporary_scripts')


models_to_be_used_small = [
    "distilbert-base-multilingual-cased",
    "microsoft/Multilingual-MiniLM-L12-H384"
    ]
models_to_be_used_base = [
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base"
    ]

models_to_be_used_large = [
    "xlm-roberta-large"
    ]



task_code_mapping = {
    'run_greek_legal_ner': 'run_greek_legal_ner.py',
    'run_covid19_emergency_event': 'run_covid19_emergency_event.py',
    'run_brazilian_court_decisions_unanimity': 'run_brazilian_court_decisions_unanimity.py',
    'run_greek_legal_code_chapter_level': 'run_greek_legal_code_chapter_level.py',
    'run_online_terms_of_service_unfairness_category': 'run_online_terms_of_service_unfairness_category.py',
    'run_mapa_ner_fine_grained': 'run_mapa_ner_fine_grained.py',
    'run_multi_eurlex_level_1': 'run_multi_eurlex_level_1.py',
    'run_multi_eurlex_level_2': 'run_multi_eurlex_level_2.py',
    'run_multi_eurlex_level_3': 'run_multi_eurlex_level_3.py',
    'run_greek_legal_code_volume_level': 'run_greek_legal_code_volume_level.py',
    'run_online_terms_of_service_unfairness_level': 'run_online_terms_of_service_unfairness_level.py',
    'run_brazilian_court_decisions_judgment': 'run_brazilian_court_decisions_judgment.py',
    'run_legalnero': 'run_legalnero.py',
    'run_german_argument_mining': 'run_german_argument_mining.py',
    'run_mapa_ner_coarse_grained': 'run_mapa_ner_coarse_grained.py',
    'run_swiss_judgment_prediction': 'run_swiss_judgment_prediction.py',
    'run_lener_br': 'run_lener_br.py',
    'run_greek_legal_code_subject_level': 'run_greek_legal_code_subject_level.py'
    }


def generate_command(time_stamp, **data):

    time_now = datetime.datetime.now().isoformat().split(':')[:1][0]

    command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} python ./experiments/{CODE} --model_name_or_path {MODEL_NAME} --do_lower_case {LOWER_CASE}  --output_dir results/logs'+time_stamp+'/'+'{TASK}/{MODEL_NAME}/seed_{SEED} --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model {METRIC_FOR_BEST_MODEL} --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs {NUM_TRAIN_EPOCHS} --learning_rate {LEARNING_RATE} --per_device_train_batch_size {BATCH_SIZE} --per_device_eval_batch_size {BATCH_SIZE} --seed {SEED} --fp16 --fp16_full_eval --gradient_accumulation_steps {ACCUMULATION_STEPS} --eval_accumulation_steps {ACCUMULATION_STEPS} --running_mode {RUNNING_MODE}'

    
    final_command = command_template.format(GPU_NUMBER=data["gpu_number"],MODEL_NAME=data["model_name"],LOWER_CASE=data["lower_case"],TASK=data["task"],SEED=data["seed"],NUM_TRAIN_EPOCHS=data["num_train_epochs"],BATCH_SIZE=data["batch_size"],ACCUMULATION_STEPS=data["accumulation_steps"],LANGUAGE=data["language"],RUNNING_MODE=data["running_mode"],LEARNING_RATE=data["learning_rate"],CODE=data["code"],METRIC_FOR_BEST_MODEL=data["metric_for_best_model"])

    if "hierarchical" in data.keys():
        final_command += ' --hierarchical '+data["hierarchical"]
    
    file_name = './temporary_scripts/'+data["task"]+"_"+str(data["gpu_number"])+"_"+str(data["seed"])+"_"+str(data["model_name"]).replace('/','_')+"_"+time_now+".sh"
    
    with open(file_name,"w") as f:
        print(final_command,file=f)
     
    return file_name

def get_optimal_batch_size(language_model:str, hierarchical:bool):


    if str(hierarchical).lower()=="false":

        if language_model=="distilbert-base-multilingual-cased":
            batch_size= 64
            accumulation_steps=1
        elif language_model=="microsoft/Multilingual-MiniLM-L12-H384":
            batch_size= 32
            accumulation_steps=2
        elif language_model=="xlm-roberta-base":
            batch_size= 32
            accumulation_steps=2
        elif language_model=="microsoft/mdeberta-v3-base":
            batch_size= 16
            accumulation_steps=4
        elif language_model=="xlm-roberta-large":
            batch_size= 16
            accumulation_steps=4
        
        return batch_size, accumulation_steps

    elif str(hierarchical).lower()=="true":

        if language_model=="distilbert-base-multilingual-cased":
            batch_size= 16
            accumulation_steps=4
        elif language_model=="microsoft/Multilingual-MiniLM-L12-H384":
            batch_size= 16
            accumulation_steps=4
        elif language_model=="xlm-roberta-base":
            batch_size= 8
            accumulation_steps=8
        elif language_model=="microsoft/mdeberta-v3-base":
            batch_size= 4
            accumulation_steps=16
        elif language_model=="xlm-roberta-large":
            batch_size= 4
            accumulation_steps=16
        
        return batch_size, accumulation_steps
    
    

    
def run_script(command):
    try:
        os.system(command=command)
    except KeyboardInterrupt:
        print ('Process interrupted.')


def run_in_parallel(commands_to_run):
    if len(commands_to_run)>0:
        pool = Pool(processes=len(commands_to_run))
        pool.map(run_script, commands_to_run)

def run_experiment(running_mode,language_model_type, task,list_of_seeds,num_train_epochs,batch_size,accumulation_steps,lower_case,language,learning_rate,gpu_number,hierarchical):

    time_stamp = datetime.datetime.now().isoformat()

    if gpu_number is None:
        gpu_number = [n for n in range(0,torch.cuda.device_count())]
    else:
        gpu_number = [gpu_number]

    #Tagging my Python scripts
    setproctitle.setproctitle('Veton Matoshi - LEXTREME')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpu_number])

    if type(list_of_seeds)==list:
        list_of_seeds = [int(s) for s in list_of_seeds if s]
    elif list_of_seeds is None:
        list_of_seeds=[1,2,3,4,5]
    elif type(list_of_seeds)==str:
        list_of_seeds = list_of_seeds.split(',')
        list_of_seeds = [int(s) for s in list_of_seeds if s]
   
    print(list_of_seeds)    

    
    if language_model_type=='all':
        models_to_be_used = models_to_be_used_small + models_to_be_used_base + models_to_be_used_large

    elif language_model_type=='small':
        models_to_be_used = models_to_be_used_small

    elif language_model_type=='base':
        models_to_be_used = models_to_be_used_base

    elif language_model_type=='large':
        models_to_be_used = models_to_be_used_large
    else:
        models_to_be_used = [language_model_type]

    models_to_be_used = sorted(list(set(models_to_be_used)))
    print(models_to_be_used)

    gpu_command_dict = defaultdict(list)
    commands_already_satisfied = list()
    all_commands_to_run = list()
    number_of_parallel_commands = len(gpu_number)-1
    commands_to_run = list()
    if task=='all': 
        all_variables = [[t for t in list(task_code_mapping.keys())],models_to_be_used,list_of_seeds]
        all_variables_perturbations = list(itertools.product(*all_variables))
        all_variables_perturbations = ['$'.join([str(x) for x in p]) for p in all_variables_perturbations]        
        all_variables_perturbations = list(zip(cycle(gpu_number), all_variables_perturbations)) 
        all_variables_perturbations = [[x[0]]+x[1].split('$') for x in all_variables_perturbations]
        print(all_variables_perturbations)
             
        for (gpu_id,task,model_name,seed) in all_variables_perturbations:
            if task in ["run_greek_legal_ner", "run_mapa_ner_fine_grained", "run_mapa_ner_coarse_grained", "run_legalnero", "run_lener_br"]:
                metric_for_best_model="macro-f1"
            elif task in ["run_covid19_emergency_event", "run_multi_eurlex", "run_online_terms_of_service_unfairness_category"]:
                metric_for_best_model="macro-f1"
            else:
                metric_for_best_model="mcc"
            gpu_id = int(gpu_id)
            seed = int(seed)
            if batch_size is None:
                batch_size, accumulation_steps = get_optimal_batch_size(model_name,hierarchical)
            else:
                if accumulation_steps is None:
                    accumulation_steps = 1
            script_new = generate_command(time_stamp=time_stamp,gpu_number=gpu_id,model_name=model_name,lower_case=lower_case,task=task,seed=seed,num_train_epochs=num_train_epochs,batch_size=batch_size,accumulation_steps=accumulation_steps,language=language,running_mode=running_mode,learning_rate=learning_rate,code=task_code_mapping[task],metric_for_best_model=metric_for_best_model,hierarchical=hierarchical)
            if script_new is not None:
                command = 'bash '+str(script_new)
                gpu_command_dict[gpu_id].append(command)
                print(command)
                if command not in commands_to_run:
                    if len(commands_to_run) <= number_of_parallel_commands:
                        commands_to_run.append(command)
                        commands_already_satisfied.append((model_name,task,seed))
                    else:
                        all_commands_to_run.append(commands_to_run)
                        commands_to_run=list()
                        commands_to_run.append(command)
                        commands_already_satisfied.append((model_name,task,seed))
    else:
        all_variables = [[task],models_to_be_used,list_of_seeds]
        all_variables_perturbations = list(itertools.product(*all_variables))
        all_variables_perturbations = ['$'.join([str(x) for x in p]) for p in all_variables_perturbations]        
        all_variables_perturbations = list(zip(cycle(gpu_number), all_variables_perturbations)) 
        all_variables_perturbations = [[x[0]]+x[1].split('$') for x in all_variables_perturbations]
             
        for (gpu_id,task,model_name,seed) in all_variables_perturbations:
            if task in ["run_greek_legal_ner", "run_mapa_ner_fine_grained", "run_mapa_ner_coarse_grained", "run_legalnero", "run_lener_br", "run_covid19_emergency_event", "run_multi_eurlex", "run_online_terms_of_service_unfairness_category"]:
                metric_for_best_model="macro-f1"
            else:
                metric_for_best_model="mcc"
            gpu_id = int(gpu_id)
            seed = int(seed)
            if batch_size is None:
                batch_size, accumulation_steps = get_optimal_batch_size(model_name,hierarchical)
            else:
                if accumulation_steps is None:
                    accumulation_steps = 1
            script_new = generate_command(time_stamp=time_stamp,gpu_number=gpu_id,model_name=model_name,lower_case=lower_case,task=task,seed=seed,num_train_epochs=num_train_epochs,batch_size=batch_size,accumulation_steps=accumulation_steps,language=language,running_mode=running_mode,learning_rate=learning_rate,code=task_code_mapping[task],metric_for_best_model=metric_for_best_model,hierarchical=hierarchical)
            if script_new is not None:
                command = 'bash '+str(script_new)
                gpu_command_dict[gpu_id].append(command)
                print(command)
                if command not in commands_to_run:
                    if len(commands_to_run) <= number_of_parallel_commands:
                        commands_to_run.append(command)
                        commands_already_satisfied.append((model_name,task,seed))
                    else:
                        all_commands_to_run.append(commands_to_run)
                        commands_to_run=list()
                        commands_to_run.append(command)
                        commands_already_satisfied.append((model_name,task,seed))
            
    if len(all_commands_to_run)==0:
        all_commands_to_run.append(commands_to_run)

    command_dict= dict()
    for n,c in enumerate(all_commands_to_run):
        command_dict[n]=c
    
    with open('command_dict.json','w') as f:
        js.dump(command_dict,f,ensure_ascii=False,indent=2)

        
    commands_to_run = ['; sleep 10; '.join(commands) for gpu_id, commands in gpu_command_dict.items()]
    run_in_parallel(commands_to_run=commands_to_run)





if __name__=='__main__':

    

    parser = argparse.ArgumentParser()

    parser.add_argument('-as','--accumulation_steps', help='Define the number of accumulation_steps.', default=None)
    parser.add_argument('-bz','--batch_size', help='Define the batch size.', default=None)
    parser.add_argument('-gn','--gpu_number', help='Define which gpu you would like to use.', default=None)
    parser.add_argument('-hier','--hierarchical', help='Define whether you want to use a hierarchical model or not. Caution: this will not work for every task', default=None)
    parser.add_argument('-lang','--language', help='Define if you want to filter the training dataset by language.', default='all_languages')
    parser.add_argument('-lc','--lower_case', help='Define if lower case or not.', default=False)
    parser.add_argument('-lmt','--language_model_type', help='Define which kind of language model you would like to use; you can choose between small, base and large language models or all of them.', default='all')
    parser.add_argument('-los','--list_of_seeds', help='Define the number of training epochs.', default=None)
    parser.add_argument('-lr','--learning_rate', help='Define the learning rate', default=1e-5)
    parser.add_argument('-nte','--num_train_epochs', help='Define the number of training epochs.', default=50)
    parser.add_argument('-rmo','--running_mode', help='Define whether you want to run the finetungin on all available training data or just a small portion for testing purposes.', default='default')
    parser.add_argument('-t','--task', help='Choose a task.', default='all',choices=sorted(list(task_code_mapping.keys())))
    

    args = parser.parse_args()
    
    

    run_experiment(
                    accumulation_steps=args.accumulation_steps, 
                    batch_size=args.batch_size,
                    gpu_number=args.gpu_number,
                    hierarchical=args.hierarchical,
                    language=args.language, 
                    language_model_type=args.language_model_type,
                    learning_rate=args.learning_rate,
                    list_of_seeds=args.list_of_seeds,
                    lower_case=args.lower_case, 
                    num_train_epochs=args.num_train_epochs,
                    running_mode=args.running_mode,
                    task=args.task,
                    )

    
