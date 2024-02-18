from gvbench.utils import parse_pairs_from_retrieval
import numpy as np
import csv
from gvbench import logger


def compute_label(gt_list, query, reference):
    '''
        input: gt_list, query_num, reference_num
        return: 1/0
    '''
    if abs(int(query) - int(reference)) < 3:
          return 1
    else:
          return 0
#     if reference in gt_list[query]:
#         return 1
#     else:
#         return 0


def compute_label_list(pairs_path, gt_path, label_path):
    '''
        input: pairs, gt_list
        return: label_list
    '''
    fout = open(label_path, 'w')
    
#     gt_list = np.load(gt_path, allow_pickle=True)
    pairs_loader = parse_pairs_from_retrieval(pairs_path)
    for query, reference in pairs_loader:
        q_num = int(query.split('/')[-1].split('.')[0].split('image')[-1])
        r_num = int(reference.split('/')[-1].split('.')[0].split('image')[-1])
        logger.debug(f'query: {q_num}, reference: {r_num}')
        label = compute_label(None, q_num, r_num)
        fout.write(f'{query} {reference} {label}\n')
    fout.close()
    

# def compute_gt_list(csv_path: str):
#       with open(csv_path, 'r') as csv_file:
#             csv_reader = csv.reader(csv_file, delimiter=',')
#             line_count = 0
#             gt_list = []
#             for row in csv_reader:
#                   if line_count == 0:
#                         logger.info(f'Column names are {", ".join(row)}')
#                         line_count += 1
#                   else:
#                         gt_list.append
#             print(f'Processed {line_count} lines.')
      
      
            


if __name__ == "__main__":
      pairs_path = "dataset/Nordland_RAS2020/cosplace_pairs_10.txt"
      # gt_path = "dataset/Nordland/ground_truth_new.npy"
      label_path = "dataset/Nordland_RAS2020/cosplace_pairs_10_gt2.txt"
      compute_label_list(pairs_path, None, label_path)