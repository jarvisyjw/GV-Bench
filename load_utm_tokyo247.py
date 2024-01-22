import pandas as pd

csv_path = 'dataset/tokyo247/images/query/00001.csv'
a = pd.read_csv(csv_path)
# print(a)
# print(a[2], a[1])
output_list = [i for i in a]
print(output_list[-2], output_list[-1])
# csv_path = 'dataset/tokyo247/images/query/00001.csv'