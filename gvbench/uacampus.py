import numpy as np
import scipy
from .utils import parse_pairs_from_retrieval

def load_gt(gt_file):
    gt = scipy.io.loadmat(gt_file)
    gt = gt["UAcampus"]
    gt = np.array(gt, dtype=np.uint)
    img_nums = 647
    gt = gt[:img_nums, :img_nums]

    gt_list=[]  # (img_nums, [1,2,3,...])
    for i in range(img_nums):
        gt_list.append(np.where(gt[i]==1)[0].tolist())
    return gt_list


def compute_label(gt_list, query, reference):
    '''
        input: gt_list, query_num, reference_num
        return: 1/0
    '''
    if reference in gt_list[query]:
        return 1
    else:
        return 0


def compute_label_list(pairs_path, gt_list, label_path):
    '''
        input: pairs, gt_list
        return: label_list
    '''
    fout = open(label_path, 'w')
    
    pairs_loader = parse_pairs_from_retrieval(pairs_path)
    for query, reference in pairs_loader:
        q_num = int(query.split('/')[-1].split('.')[0])
        r_num = int(reference.split('/')[-1].split('.')[0])
        label = compute_label(gt_list, q_num, r_num)
        fout.write(f'{query} {reference} {label}\n')
    fout.close()


if __name__ == "__main__":
    gt_file = "dataset/uacampus/gt.mat"
    gt_list = load_gt(gt_file)
    pairs_path = "dataset/uacampus/pairs_netvlad.txt"
    label_path = "dataset/uacampus/pairs_gt.txt"
    compute_label_list(pairs_path, gt_list, label_path)

