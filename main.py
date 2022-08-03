import os
import argparse
from multiprocessing import Pool
import torch
from itertools import cycle
import datetime
import json as js
import shutil
import itertools
import setproctitle


os.chdir('./scripts')

#Empty folder with temporary scripts
shutil.rmtree('./temporary', ignore_errors=True)
os.mkdir('./temporary')


models_to_be_used_small = open('models_to_be_used_small.txt','r').readlines()
models_to_be_used_small = [x.strip() for x in models_to_be_used_small]
models_to_be_used_base = open('models_to_be_used_base.txt','r').readlines()
models_to_be_used_base = [x.strip() for x in models_to_be_used_base]
models_to_be_used_large = open('models_to_be_used_large.txt','r').readlines()
models_to_be_used_large = [x.strip() for x in models_to_be_used_large]


task_code_mapping = {'run_greek_legal_ner': 'run_greek_legal_ner.py',
    'run_covid19_emergency_event': 'run_covid19_emergency_event.py',
    'run_brazilian_court_decisions_unanimity': 'run_brazilian_court_decisions_unanimity.py',
    'run_greek_legal_code_chapter_level': 'run_greek_legal_code_chapter_level.py',
    'run_online_terms_of_service_unfairness_category': 'run_online_terms_of_service_unfairness_category.py',
    'run_mapa_ner_fine_grained': 'run_mapa_ner_fine_grained.py',
    'run_multi_eurlex': 'run_multi_eurlex.py',
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


def generate_command(**data):

    time_now = datetime.datetime.now().isoformat().split(':')[:1][0]

    command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} python ../experiments/{CODE} --model_name_or_path {MODEL_NAME} --do_lower_case {LOWER_CASE}  --output_dir logs/{LANGUAGE}'+'_'+'{TASK}/{MODEL_NAME}/seed_{SEED} --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model {METRIC_FOR_BEST_MODEL} --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs {NUM_TRAIN_EPOCHS} --learning_rate {LEARNING_RATE} --per_device_train_batch_size {BATCH_SIZE} --per_device_eval_batch_size {BATCH_SIZE} --seed {SEED} --fp16 --fp16_full_eval --gradient_accumulation_steps {ACCUMULATION_STEPS} --eval_accumulation_steps {ACCUMULATION_STEPS} --running_mode {RUNNING_MODE}'

    
    final_command = command_template.format(GPU_NUMBER=data["gpu_number"],MODEL_NAME=data["model_name"],LOWER_CASE=data["lower_case"],TASK=data["task"],SEED=data["seed"],NUM_TRAIN_EPOCHS=data["num_train_epochs"],BATCH_SIZE=data["batch_size"],ACCUMULATION_STEPS=data["accumulation_steps"],LANGUAGE=data["language"],RUNNING_MODE=data["running_mode"],LEARNING_RATE=data["learning_rate"],CODE=data["code"],METRIC_FOR_BEST_MODEL=data["metric_for_best_model"]) # I must be careful, it another stragey might require greater_is_better = False
    
    file_name = './temporary/'+data["task"]+"_"+str(data["gpu_number"])+"_"+str(data["seed"])+"_"+str(data["model_name"]).replace('/','_')+"_"+time_now+".sh"
    with open(file_name,"w") as f:
        print(final_command,file=f)
     
    return file_name
    
def run_script(command):
    os.system(command=command)


def run_in_parallel(commands_to_run):
    if len(commands_to_run)>0:
        pool = Pool(processes=len(commands_to_run))
        pool.map(run_script, commands_to_run)

def run_experiment(language_model_type='all',running_mode='default', task='all',list_of_seeds=[1,2,3,4],lower_case=True,num_train_epochs=20,batch_size=10,accumulation_steps=1,language='all_languages',learning_rate=1e-5,gpu_number=None):


    if gpu_number is None:
        gpu_number = [n for n in range(0,torch.cuda.device_count())]
    else:
        gpu_number = [gpu_number]

    #Tagging my Python scripts
    setproctitle.setproctitle('Veton Matoshi - LEXTREME')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpu_number])

    if list_of_seeds is None:
        list_of_seeds=[1,2,3,4]
    else:
        list_of_seeds = list_of_seeds.split(',')
        list_of_seeds = [int(s) for s in list_of_seeds]
   
    

    
    if language_model_type=='all':
        models_to_be_used = models_to_be_used_small + models_to_be_used_base + models_to_be_used_large

    elif language_model_type=='small':
        models_to_be_used = models_to_be_used_small

    elif language_model_type=='base':
        models_to_be_used = models_to_be_used_base

    elif language_model_type=='large':
        models_to_be_used = models_to_be_used_large

    models_to_be_used = list(set(models_to_be_used))
    print(models_to_be_used)

    commands_already_satisfied = list()
    all_commands_to_run = list()
    number_of_parallel_commands = len(gpu_number)-1
    commands_to_run = list()
    if task=='all': 
        all_variables = [[t for t in list(task_code_mapping.keys()) if t!='run_multi_eurlex'],models_to_be_used,list_of_seeds] #Remove multi_eur_lex and put it at the end because it takes ages to finish
        all_variables_perturbations = list(itertools.product(*all_variables))
        all_variables_perturbations = ['$'.join([str(x) for x in p]) for p in all_variables_perturbations]        
        all_variables_perturbations = list(zip(cycle(gpu_number), all_variables_perturbations)) 
        all_variables_perturbations = [[x[0]]+x[1].split('$') for x in all_variables_perturbations]
             
        for (gpu_id,task,model_name,seed) in all_variables_perturbations:
            if task in ["run_greek_legal_ner", "run_mapa_ner_fine_grained", "run_mapa_ner_coarse_grained", "run_legalnero", "run_lener_br"]:
                metric_for_best_model="overall_f1"
            else:
                metric_for_best_model="micro-f1"
            gpu_id = int(gpu_id)
            seed = int(seed)
            script_new = generate_command(gpu_number=gpu_id,model_name=model_name,lower_case=lower_case,task=task,seed=seed,num_train_epochs=num_train_epochs,batch_size=batch_size,accumulation_steps=accumulation_steps,language=language,running_mode=running_mode,learning_rate=learning_rate,code=task_code_mapping[task],metric_for_best_model=metric_for_best_model)
            if script_new is not None:
                command = 'bash '+str(script_new)
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
            if task in ["run_greek_legal_ner", "run_mapa_ner_fine_grained", "run_mapa_ner_coarse_grained", "run_legalnero", "run_lener_br"]:
                metric_for_best_model="overall_f1"
            else:
                metric_for_best_model="micro-f1"
            gpu_id = int(gpu_id)
            seed = int(seed)
            script_new = generate_command(gpu_number=gpu_id,model_name=model_name,lower_case=lower_case,task=task,seed=seed,num_train_epochs=num_train_epochs,batch_size=batch_size,accumulation_steps=accumulation_steps,language=language,running_mode=running_mode,learning_rate=learning_rate,code=task_code_mapping[task],metric_for_best_model=metric_for_best_model)
            if script_new is not None:
                command = 'bash '+str(script_new)
                if command not in commands_to_run:
                    if len(commands_to_run) <= number_of_parallel_commands:
                        commands_to_run.append(command)
                        commands_already_satisfied.append((model_name,task,seed))
                    else:
                        all_commands_to_run.append(commands_to_run)
                        commands_to_run=list()
                        commands_to_run.append(command)
                        commands_already_satisfied.append((model_name,task,seed))
            
    
    command_dict= dict()
    for n,c in enumerate(all_commands_to_run):
        command_dict[n]=c
    
    with open('command_dict.json','w') as f:
        js.dump(command_dict,f,ensure_ascii=False,indent=2)

            
            
    for commands_to_run in all_commands_to_run:    
        print(commands_to_run)
        run_in_parallel(commands_to_run)

   
    
            


            


            

           




if __name__=='__main__':

    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lmt','--language_model_type', help='Define which kind of language model you would like to use; you can choose between small, base and large language models or all of them.', default='all')
    parser.add_argument('-rmo','--running_mode', help='Define whether you want to run the finetungin on all available training data or just a small portion for testing purposes.', default='default')
    parser.add_argument('-t','--task', help='Choose a task.', default='all',choices=list(task_code_mapping.keys()))
    parser.add_argument('-bz','--batch_size', help='Define the batch size.', default=10)
    parser.add_argument('-nte','--num_train_epochs', help='Define the number of training epochs.', default=20)
    parser.add_argument('-los','--list_of_seeds', help='Define the number of training epochs.', type=str)
    

    args = parser.parse_args()
    
    

    run_experiment(args.language_model_type,args.running_mode,args.task,batch_size=args.batch_size,num_train_epochs=args.num_train_epochs,list_of_seeds=args.list_of_seeds)

    
