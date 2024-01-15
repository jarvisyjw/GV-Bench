from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from scipy.special import softmax
import h5py
from typing import Tuple
import cv2
import multiprocessing


from . import logger
from .utils import parse_pairs


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


def eval_from_pair(pair: Tuple[str, str, int],
                   match_path: Path,
                   feature_path: Path, feature_path_r = None,
                   fout = None,
                   return_all = False, allow_label = False):
      
      query, reference, label = pair
      points_all, inliers = RANSACwithF(query, reference, match_path, feature_path, feature_path_r, fout)
      
      if return_all:
            if allow_label:
                  return inliers, points_all, label
            else:
                  return inliers, points_all
      else:
            if allow_label:
                  return inliers, label
            else:
                  return inliers


def eval_from_pairs(    queue: multiprocessing.Queue,
                        pairs: list,
                        match_path: Path,
                        feature_path: Path, feature_path_r = None):
      
      inliers_list = []
      labels = []
      
      if feature_path_r is None:
            feature_path_r = feature_path
      
      for pair in tqdm(pairs, total=len(pairs)):
                  inliers, label = eval_from_pair(pair, match_path, feature_path, feature_path_r, None, False, True)
                  inliers_list.append(len(inliers))
                  labels.append(label)
                  
      queue.put([inliers_list, labels])
      # return inliers_list, labels
      

def eval_from_path(gt_file_path: Path, 
                   match_path: Path, 
                   feature_path: Path, feature_path_r = None, 
                   export_dir = None,
                   return_all = False):

    pairs_loader = parse_pairs(str(gt_file_path), allow_label=True)
    pairs = [(q, r, int(l)) for q, r, l in pairs_loader]
    labels = [pair[2] for pair in pairs]
    
    if feature_path_r is None:
          feature_path_r = feature_path
    
    if export_dir is not None:
          if not Path(export_dir).exists():
                Path(export_dir).mkdir(parents=True, exist_ok=True)
                
          file_out = Path(export_dir, 'eval_log.txt')
          fout = open(str(file_out), 'w')
          out_log = Path(export_dir, f'{gt_file_path.stem}_log.npy')
      
    else:
          fout = None
          out_log  = None
    
    inliers_list = []
    points_all_list = []
    
    logger.info(f'Evaluating {len(pairs)} pairs')
    
    for pair in tqdm(pairs, total=len(pairs)):
        logger.debug(f'pair: {pair}')
        if return_all:
                  inliers, points_all = eval_from_pair(pair, match_path, feature_path, feature_path_r, fout, return_all)
                  inliers_list.append(len(inliers))
                  points_all_list.append(points_all)
        else:
              inliers = eval_from_pair(pair, match_path, feature_path, feature_path_r, fout, return_all)
              inliers_list.append(len(inliers)) 
    
    matches_pts = np.array(inliers_list)
    logger.debug(f'matches_pts: {matches_pts}')
    logger.debug(f'Max num of matches is {max(matches_pts)}')
    matches_pts_norm = matches_pts / max(matches_pts)
    average_precision = average_precision_score(labels, matches_pts_norm)
    precision, recall, TH = precision_recall_curve(labels, matches_pts_norm)
    
    if out_log is not None:
          np.save(str(out_log), {'prob': matches_pts_norm, 'gt': labels})
    
    if return_all:
          return precision, recall, average_precision, inliers_list, points_all_list
    
    return precision, recall, average_precision, inliers_list


def eval_from_path_multiprocess(num_process: int,
                                gt_file_path: Path, 
                                match_path: Path, 
                                feature_path: Path, feature_path_r = None, 
                                export_dir = None,
                                return_all = False):
      
      if feature_path_r is None:
            feature_path_r = feature_path
      
      if export_dir is not None:
            if not Path(export_dir).exists():
                  Path(export_dir).mkdir(parents=True, exist_ok=True)
                  
            file_out = Path(export_dir, 'eval_log.txt')
            fout = open(str(file_out), 'w')
            out_log = Path(export_dir, f'{gt_file_path.stem}_log.npy')
            
      else:
            fout = None
            out_log  = None
      
      inliers_list = []
      points_all_list = []
            
      logger.info(f'Using {num_process} processes for acceleration.')
      q = multiprocessing.Queue()
      
      num_matches = []
      labels = []
      
      pairs_loader = parse_pairs(str(gt_file_path), allow_label=True)
      pairs = [(q, r, int(l)) for q, r, l in pairs_loader]
      logger.info(f'Evaluating {len(pairs)} pairs')
      
      pairs_list = [pairs[i::num_process] for i in range(num_process)]
      
      processes = []
      rets = []
      
     
      for i in range(num_process):
            p = multiprocessing.Process(target=eval_from_pairs, args=(q, pairs_list[i], match_path, feature_path, feature_path_r))
            processes.append(p)
            p.start()
      
      for p in processes:
            ret = q.get()
            # logger.info(f'ret: {ret}')
            # rets.extend(ret)
            num_matches.extend(ret[0])
            labels.extend(ret[1])
      
      for p in processes:
            p.join()
      
      
      # for ret in rets:
      #       logger.debug(f'ret: {ret}')
      #       num_matches.extend(ret[0])
      #       labels.extend(ret[1])

      logger.debug(f'num_matches: {num_matches}')
      logger.debug(f'labels: {labels}')
      
      
      matches_pts = np.array(num_matches)
      labels = np.array(labels)
      
      logger.debug(f'matches_pts: {matches_pts}')
      logger.debug(f'Max num of matches is {max(matches_pts)}')
      matches_pts_norm = matches_pts / max(matches_pts)
      average_precision = average_precision_score(labels, matches_pts_norm)
      precision, recall, TH = precision_recall_curve(labels, matches_pts_norm)
      
      if out_log is not None:
            np.save(str(out_log), {'prob': matches_pts_norm, 'gt': labels})
      
      if return_all:
            return precision, recall, average_precision, inliers_list, points_all_list
      
      return precision, recall, average_precision, inliers_list
      


if __name__ == '__main__':
    
    gt_file_path = f'dataset/robotcar/gt/robotcar_qAutumn_dbSuncloud.txt'
    match_path = Path('dataset/robotcar/matches/robotcar_qAutumn_dbSuncloud/matches-NN-mutual-ratio.8.h5')
    feature_path = Path('dataset/robotcar/features/sift.h5')
    precision, recall, average_precision, inliers_list = eval_from_path_multiprocess(80, gt_file_path, match_path, feature_path)
    plot_pr_curve(recall, precision, average_precision, 'robotcar', 'sift')
    _, r_recall = max_recall(precision, recall)

    logger.info(f'\n' +
                f'Evaluation results: \n' +
                'Average Precision: {:.3f} \n'.format(average_precision) + 
                'Maximum Recall @ 100% Precision: {:.3f} \n'.format(r_recall))
    output_path = Path(f'dataset/robotcar/exps/qAutumn_dbSuncloud/sift_NN/pr_curve.png')
    if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path))
    
    # for i in range(4):
    #     gt_file_path = f'dataset/robotcar/gt/robotcar_qAutumn_dbSuncloud_dist_{i}.txt'
    #     match_path = Path('dataset/robotcar/matches/robotcar_qAutumn_dbSuncloud/matches-NN-mutual-ratio.8.h5')
    #     feature_path = Path('dataset/robotcar/features/sift.h5')
    #     precision, recall, average_precision, inliers_list = eval_from_path_multiprocess(80, gt_file_path, match_path, feature_path)
    #     plot_pr_curve(recall, precision, average_precision, 'robotcar', 'sift')
    #     _, r_recall = max_recall(precision, recall)

    #     logger.info(f'\n' +
    #                 f'Evaluation results: \n' +
    #                 'Average Precision: {:.3f} \n'.format(average_precision) + 
    #                 'Maximum Recall @ 100% Precision: {:.3f} \n'.format(r_recall))
    #     output_path = Path(f'dataset/robotcar/exps/qAutumn_dbSuncloud/sift_NN/pr_curve_dist_{i}.png')
    #     if not output_path.parent.exists():
    #             output_path.parent.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(str(output_path))
    #     plt.clf()
      
