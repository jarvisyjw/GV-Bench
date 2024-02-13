from .utils import parse_pairs_from_retrieval
import numpy as np
from . import logger


def compute_label(gt_list, query, reference):
    '''
        input: gt_list, query_num, reference_num
        return: 1/0
    '''
    if reference in gt_list[query]:
        return 1
    else:
        return 0


def compute_label_list(pairs_path, gt_path, label_path):
    '''
        input: pairs, gt_list
        return: label_list
    '''
    fout = open(label_path, 'w')
    
    gt_list = np.load(gt_path, allow_pickle=True)
    pairs_loader = parse_pairs_from_retrieval(pairs_path)
    for query, reference in pairs_loader:
        q_num = int(query.split('/')[-1].split('.')[0])
        r_num = int(reference.split('/')[-1].split('.')[0])
        logger.debug(f'query: {q_num}, reference: {r_num}')
        label = compute_label(gt_list, q_num, r_num)
        fout.write(f'{query} {reference} {label}\n')
    fout.close()


if __name__ == "__main__":
      pairs_path = "dataset/Nordland/pairs_cosplace.txt"
      gt_path = "dataset/Nordland/ground_truth_new.npy"
      label_path = "dataset/Nordland/pairs_cosplace_gt.txt"
      compute_label_list(pairs_path, gt_path, label_path)