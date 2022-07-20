from pathlib import Path
import json as js
import pandas as pd



def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result



log_path = Path('../logs/')

#directories = [x[0] for x in os.walk(log_path)]
directories = [f for f in log_path.glob('**/*') if f.is_dir()]
print(directories)

overview = list()

for d in directories:
    all_files = [f for f in d.glob('**/*') if f.is_file()]
    for f in all_files:
        if str(f).endswith('all_results.json'):
            with open(f,'r') as f:
                all_results = js.load(f)
        if str(f).endswith('config.json'):
            with open(f,'r') as f:
                config = js.load(f)
            all_results = merge_dicts(all_results, config)
            overview.append(all_results)


overview_df = pd.DataFrame(overview)
print(overview_df)
overview_df.to_csv('../overview_of_results.csv',index=False)
