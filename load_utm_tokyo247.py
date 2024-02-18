import pandas as pd

csv_path = 'dataset/mapillary_sls/train_val/boston/query/postprocessed.csv'
a = pd.read_csv(csv_path, index_col = 0)
# print(a)
# print(a[2], a[1])
# qData = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col = 0)
# output_list = [i for i in a]
print(a)
print(a[a['key'] == 'Bd7m8vNyPIJaPBXLw5ghnw'])
# print(output_list[-2], output_list[-1])
# csv_path = 'dataset/tokyo247/images/query/00001.csv'