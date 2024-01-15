from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from scipy.special import softmax
import h5py
from typing import Tuple
import cv2


from . import logger


def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))

def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')

def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')
    

def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
        # print('uncertaintylistlens', len(uncertainty))
    if return_uncertainty:
        return p, uncertainty
    return p


def get_matches(path: Path, name0: str, name1: str, out=None) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    if out is not None:
        matches_score = np.column_stack((matches, scores))
        with open(out, 'w') as f:
            f.write('\n'.join(' '.join(map(str, match)) for match in matches_score))
        f.close()
    return matches, scores

# max recall @ 100% precision
def max_recall(precision: np.ndarray, recall: np.ndarray):
    recall_idx = np.argmax(recall)
    idx = np.where(precision == 1.0)
    max_recall = np.max(recall[idx])
    # logger.debug(f'max recall{max_recall}')
    recall_idx = np.array(np.where(recall == max_recall)).reshape(-1)
    recall_idx = recall_idx[precision[recall_idx]==1.0]
    # logger.debug(f'recall_idx {recall_idx}')
    r_precision, r_recall = float(np.squeeze(precision[recall_idx])), float(np.squeeze(recall[recall_idx]))
    # logger.debug(f'precision: {r_precision}, recall: {r_recall}')
    return r_precision, r_recall

def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, average_precision, dataset = 'robotcar', exp_name = 'test'):
    # calculate max f2 score, and max recall
    f_score = 2 * (precision * recall) / (precision + recall)
    f_idx = np.argmax(f_score)
    f_precision, f_recall = np.squeeze(precision[f_idx]), np.squeeze(recall[f_idx])
    r_precision, r_recall = max_recall(precision, recall)
    plt.plot(recall, precision, label="{} (AP={:.3f})".format(dataset, average_precision))
    plt.vlines(r_recall, np.min(precision), r_precision, colors='green', linestyles='dashed')
    plt.vlines(f_recall, np.min(precision), f_precision, colors='red', linestyles='dashed')
    plt.hlines(f_precision, np.min(recall), f_recall, colors='red', linestyles='dashed')
    plt.hlines(r_precision, np.min(recall), r_recall, colors='green', linestyles='dashed')
    plt.xlim(0, None)
    plt.ylim(np.min(precision), None)
    plt.text(f_recall + 0.005, f_precision + 0.005, '[{:.3f}, {:.3f}]'.format(f_recall, f_precision))
    plt.text(r_recall + 0.005, r_precision + 0.005, '[{:.3f}, {:.3f}]'.format(r_recall, r_precision))
    plt.scatter(f_recall, f_precision, marker='o', color='red', label='Max F score')
    plt.scatter(r_recall, r_precision, marker='o', color='green', label='Max Recall')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curves for {}".format(exp_name))


def RANSACwithH(match_path: Path, feature_path: Path, name0: str, name1: str, fout = None):

    matches, scores = get_matches(match_path, name0, name1)
    original_matches = len(matches)
    
    matches = matches[scores > 0.5]
    logger.debug(f'Found {len(matches)} matches between {name0} and {name1}.')
    kpts0 = get_keypoints(feature_path, name0)
    kpts1 = get_keypoints(feature_path, name1)
    logger.debug(f'kp0: {len(kpts0)}, kp1: {len(kpts1)}')
    logger.debug(f'kpts0: {kpts0}, kpts1: {kpts1}')
    point_map = []
    for match in matches:
        x1, y1, x2, y2 = kpts0[match[0]][0], kpts0[match[0]][1], kpts1[match[1]][0], kpts1[match[1]][1]
        point_map.append([x1, y1, x2, y2])
   
    pointMap = np.array(point_map, dtype=np.float32)
    if pointMap.shape[0] < 4:
        logger.warning(f'Not enough points to compute homography')
        return pointMap, pointMap
    logger.debug(f'solving homography...\n' + 
                     f'{name0} and {name1}')
    
    H, inliers_idx = cv2.findHomography(pointMap[:, :2], pointMap[:, 2:], cv2.RANSAC, 15.0)

    if H is None or inliers_idx is None:
        logger.warning(f'Failed to compute homography')
        return pointMap, pointMap

    logger.debug(f'H: {H}, inliers_idx: {inliers_idx}')
    inliers = pointMap[inliers_idx.ravel() == 1]
    logger.debug(f'original: {len(pointMap)}, inliers: {len(inliers)}')
    if fout is not None:
        fout.write(f'{name0} {name1} {original_matches} {len(matches)} {len(inliers)}\n')

    return pointMap, inliers


def RANSACwithF(name0: str, name1: str, 
                match_path: Path, 
                feature_path: Path, feature_path_r = None,
                fout = None):
    '''
    By default, 
            name0 -> query
            name1 -> reference
    '''

    matches, scores = get_matches(match_path, name0, name1)
    original_matches = len(matches)
    matches = matches[scores > 0.5]
    logger.debug(f'Found {len(matches)} matches between {name0} and {name1}.')
    
    if feature_path_r is None:
            feature_path_r = feature_path
    kpts0 = get_keypoints(feature_path, name0)
    kpts1 = get_keypoints(feature_path_r, name1)
    
    logger.debug(f'kp0: {len(kpts0)}, kp1: {len(kpts1)}')
    logger.debug(f'kpts0: {kpts0}, kpts1: {kpts1}')
    
    point_map = []
    for match in matches:
        x1, y1, x2, y2 = kpts0[match[0]][0], kpts0[match[0]][1], kpts1[match[1]][0], kpts1[match[1]][1]
        point_map.append([x1, y1, x2, y2])
        
    pointMap = np.array(point_map, dtype=np.float32)
    
    if pointMap.shape[0] < 8:
        logger.warning(f'Not enough points to compute fundamental matrix')
        return pointMap, pointMap
  
    logger.debug(f'solving Fundamental matrix...\n' + 
                     f'{name0} and {name1}')

    F, inliers_idx = cv2.findFundamentalMat(pointMap[:, :2], pointMap[:, 2:], cv2.FM_RANSAC, 3.0)
    
    if F is None or inliers_idx is None:
        logger.error(f'Failed to compute fundamental matrix')
        return pointMap, pointMap
    
    logger.debug(f'F: {F}, inliers_idx: {inliers_idx}')
    
    inliers = pointMap[inliers_idx.ravel() == 1]
    logger.debug(f'original: {len(pointMap)}, inliers: {len(inliers)}')
    if fout is not None:
        fout.write(f'{name0} {name1} {original_matches} {len(matches)} {len(inliers)}\n')

    return pointMap, inliers


def evaluate_from_pairs(pairs: list, 
                        match_path: Path, 
                        feature_path: Path, feature_path_r = None, 
                        export_dir = None):

    
    if feature_path_r is None:
          feature_path_r = feature_path
    
    if export_dir is not None:
          if not Path(export_dir).exists():
                Path(export_dir).mkdir(parents=True, exist_ok=True)
                
          file_out = Path(export_dir, 'eval_log.txt')
          fout = open(str(file_out), 'w')
    else:
          fout = None
    
    inliers_list = []
    labels = []
    
    pairs_loader = parse_pairs(pairs)
    pairs = [(q, r) for q, r, _ in pairs_loader]
    
          
    logger.info(f'Evaluating {len(pairs)} pairs')
    for pair in tqdm(pairs, total=len(pairs)):
        logger.debug(f'pair: {pair}')
        query, reference, label = pair
        points_all, inliers = RANSACwithF(query, reference, match_path, feature_path, feature_path_r, fout)
        
        





    with open(gt_file, 'r') as f:
        gt_labels = f.readlines()
    f.close()
    matches = []
    labels = []
    for gt_label in tqdm(gt_labels):
        query, reference, label = gt_label.strip('\n').split(', ')
        logger.debug(f'query: {query}, reference: {reference}, label: {label}')
        pointmap, inliers = matcher_helper.loadPointMap(query, reference, pointMap)
        # if inliers == None:
        #     logger.debug(f'original: {len(pointmap)}, inliers: {0}, label: {label}')
        #     matches.append(pointmap)
        #     labels.append(int(label))
        #     continue
        logger.debug(f'original: {len(pointmap)}, inliers: {len(inliers)}, label: {label}')
        matches.append(len(inliers))
        labels.append(int(label))
    matches_pts = np.array(matches)
    logger.debug(f'matches_pts: {matches_pts}')
    logger.debug(f'Max num of matches is {max(matches_pts)}')
    matches_pts_norm = matches_pts / max(matches_pts)
    average_precision = average_precision_score(labels, matches_pts_norm)
    precision, recall, TH = precision_recall_curve(labels, matches_pts_norm)
    plot_pr_curve(recall, precision, average_precision, dataset, exp_name)
    _, r_recall = max_recall(precision, recall)

    logger.info(f'\n' +
                f'Evaluation results: \n' +
                'Average Precision: {:.3f} \n'.format(average_precision) + 
                f'Maximum & Minimum matches: {np.max(matches_pts)}' + ' & ' + f'{np.min(matches_pts)} \n' +
                'Maximum Recall @ 100% Precision: {:.3f} \n'.format(r_recall))

# if __name__ == '__main__':
#     # val_log='/home/jarvis/jw_ws/Verification/doppelgangers/logs/oxford_Autumn_SunCloud_finetune_2023-Dec-09-13-21-54/test_doppelgangers_list.npy'
#     val_log = '/home/jarvis/jw_ws/Verification/doppelgangers/logs/oxford_Autumn_SunCloud_2023-Dec-08-18-54-40/test_doppelgangers_list.npy'
#     # val_log = '/home/jarvis/jw_ws/Verification/doppelgangers/logs/oxford_qAutumn_dbNight_val_2023-Dec-07-13-18-31/test_doppelgangers_list.npy'

#     import numpy
#     result_list = numpy.load(val_log, allow_pickle=True)
# #     import pdb; pdb.set_trace()
#     result_list = result_list.tolist()
#     pred = result_list['pred']
#     gt_list = result_list['gt']
#     prob = result_list['prob']
#     y_scores_s = softmax(prob, axis=1)
#     y_scores = y_scores_s[:, 1]
#     precision, recall, TH = precision_recall_curve(gt_list, y_scores)
#     average_precision = average_precision_score(gt_list, y_scores)
#     plot_pr_curve(recall, precision, average_precision, 'robotcar', 'doppelgangers')
#     plt.show()
# #     import pdb; pdb.set_trace()
#     # plt.scatter(recall, precision, c = np.array([*TH,1]).reshape(-1,1))
# #     plt.plot(recall, precision)
# #     plot_pr_curve()
#     # plt.show()
