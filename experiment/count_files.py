import os

path = './data'

files_data = [f.split('.')[0] for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]

path2 = './ot_result/fugw_par'
files_fugw = [f.split('.')[0] for f in os.listdir(path2)
             if os.path.isfile(os.path.join(path2, f)) and f.endswith('.json')]
res = list(set(files_data)-set(files_fugw))
print(res)
print(len(res))
