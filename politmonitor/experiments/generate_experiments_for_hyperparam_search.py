import json as js
import itertools
import argparse
import os
import sys

_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')
sys.path.append(_path)
with open(_path, 'r') as f:
    hyperparameters = js.load(f)
learning_rate = hyperparameters['learning_rate']
batch_size = hyperparameters['batch_size']
weight_decay = hyperparameters['weight_decay']

all_hyperparams = [learning_rate, batch_size, weight_decay]
all_hyperparams = list(itertools.product(*all_hyperparams))


def generate_commands(gn):
    commands = list()
    for (learning_rate, batch_size, weight_decay) in all_hyperparams:
        folder_name = 'joelito/legal-xlm-roberta-large_' + '__learning_rate_' + \
                      str(learning_rate) + '__batch_size_' + str(batch_size) + \
                      '__weight_decay_' + str(weight_decay)
        comm = 'CUDA_VISIBLE_DEVICES=' + gn + ' python3 ./experiments/template_MLTC.py --finetuning_task politmonitor --model_name_or_path joelito/legal-xlm-roberta-large --log_directory hyperparameter_search_politmonitor --preprocessing_num_workers 1 --hierarchical False --revision main --affair_text_scope zh,ch  --output_dir hyperparameter_search_politmonitor/' + \
               folder_name + ' --fp16 --fp16_full_eval --metric_for_best_model macro-f1 --greater_is_better true --do_train --do_eval --do_predict --num_train_epochs 30 --load_best_model_at_end --save_strategy epoch --logging_strategy epoch --evaluation_strategy epoch --per_device_train_batch_size '+str(batch_size) +' --per_device_eval_batch_size '+str(batch_size) +' --weight_decay '+str(weight_decay) + ' --learning_rate ' + str(learning_rate)
        commands.append(comm)

    return ' ; sleep 10 ; '.join(commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-gn')

    args = parser.parse_args()

    commands = generate_commands(args.gn)
    # print(commands)

    os.system(commands)