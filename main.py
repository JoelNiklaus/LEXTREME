import os
import argparse
from pathlib import Path
import re
from multiprocessing import Pool
import torch
from itertools import cycle
from ast import literal_eval

os.chdir('./scripts')


models_to_be_used_small = open('models_to_be_used_small.txt','r').readlines()
models_to_be_used_small = [x.strip() for x in models_to_be_used_small]
models_to_be_used_base = open('models_to_be_used_base.txt','r').readlines()
models_to_be_used_base = [x.strip() for x in models_to_be_used_base]
models_to_be_used_large = open('models_to_be_used_large.txt','r').readlines()
models_to_be_used_large = [x.strip() for x in models_to_be_used_large]




def generate_command(file_path,gpu_number,seed,batch_size=None,num_train_epochs=None):

    
    lines = list()
    with open(file_path,'r') as f:
        for line in f:
            if bool(re.search('GPU_NUMBER=',line)):
                line = 'GPU_NUMBER='+str(gpu_number)
            if bool(re.search('SEED=',line)):
                line = 'SEED='+str(seed)
            if bool(re.search('GPU_NUMBER=',line)):
                line = 'GPU_NUMBER='+str(gpu_number)
            if batch_size is not None:
                if bool(re.search('BATCH_SIZE=',line)):
                    line = 'BATCH_SIZE='+str(batch_size)
            if num_train_epochs is not None:
                if bool(re.search('NUM_TRAIN_EPOCHS=',line)):
                    line = 'NUM_TRAIN_EPOCHS='+str(num_train_epochs)
            

            lines.append(line)
    
    file_path_new = 'temporary/'+str(file_path).split('/')[-1][:-3]+'_temporary_'+str(gpu_number)+'__'+str(seed)+'.sh'
    with open(file_path_new,'w') as f:
        for line in lines:
            print(line, file=f)

    return file_path_new
    
def run_script(command):
    os.system(command=command)


def run_in_parallel(commands_to_run):
    if len(commands_to_run)>0:
        pool = Pool(processes=len(commands_to_run))
        pool.map(run_script, commands_to_run)

def run_experiment(language_model_type='all',running_mode='default', task='all',list_of_seeds=None,batch_size=None,num_train_epochs=None,gpu_number=None):

    if gpu_number is None:
        gpu_number = [n for n in range(0,torch.cuda.device_count())]
    else:
        gpu_number = [gpu_number]

    if list_of_seeds is None:
        list_of_seeds=[1,2,3,4]
    else:
        list_of_seeds = list_of_seeds.split(',')
        list_of_seeds = [int(s) for s in list_of_seeds]

    gpu_seed_combinations = list(zip(cycle(gpu_number), list_of_seeds))
    n = len(gpu_number)
    gpu_seed_combinations_chunks = [gpu_seed_combinations[x:x+n] for x in range(0, len(gpu_seed_combinations), n)]

    
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
        
        if str(script).endswith('sh') and 'temporary' not in str(script):
            if task=='all':
                print(script)
                commands_to_run = list()
                for command_chunks in gpu_seed_combinations_chunks:
                    for (gpu_number, seed) in command_chunks:
                        print(gpu_number,seed)
                        script_new = generate_command(script,gpu_number=gpu_number,seed=seed,batch_size=batch_size,num_train_epochs=num_train_epochs)
                        command = 'bash '+str(script_new)
                        commands_to_run.append(command)
                
                print(commands_to_run)
                run_in_parallel(commands_to_run)

            else:
                if bool(re.search(str(task),str(script)))==True:
                    print(script)
                    commands_to_run = list()
                    for command_chunks in gpu_seed_combinations_chunks:
                        for (gpu_number, seed) in command_chunks:
                            print(gpu_number,seed)
                            script_new = generate_command(script,gpu_number=gpu_number,seed=seed,batch_size=batch_size,num_train_epochs=num_train_epochs)
                            command = 'bash '+str(script_new)
                            commands_to_run.append(command)
                    print(commands_to_run)
                    run_in_parallel(commands_to_run)
        


            


            

           




if __name__=='__main__':

    list_of_tasks = os.listdir('./')
    list_of_tasks = [x.strip()[:-3] for x in list_of_tasks if x.endswith('sh')]
    list_of_tasks = [re.sub(r'run_','',x) for x in list_of_tasks]
    list_of_tasks = sorted(list_of_tasks)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lmt','--language_model_type', help='Define which kind of language model you would like to use; you can choose between small, base and large language models or all of them.', default='all')
    parser.add_argument('-rmo','--running_mode', help='Define whether you want to run the finetungin on all available training data or just a small portion for testing purposes.', default='default')
    parser.add_argument('-t','--task', help='Choose a task.', default='all',choices=list_of_tasks)
    parser.add_argument('-bz','--batch_size', help='Define the batch size.', default=10)
    parser.add_argument('-nte','--num_train_epochs', help='Define the number of training epochs.', default=20)
    parser.add_argument('-los','--list_of_seeds', help='Define the number of training epochs.', type=str)
    

    args = parser.parse_args()
    
    

    run_experiment(args.language_model_type,args.running_mode,args.task,batch_size=args.batch_size,num_train_epochs=args.num_train_epochs,list_of_seeds=args.list_of_seeds)

