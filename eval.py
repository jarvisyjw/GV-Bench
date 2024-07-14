import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
from typing import Tuple
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import Dataset, DataLoader

from utils import logger, get_matches, get_keypoints, parse_pairs


def inmask(x, y, mask):
    if x > mask[0] and y > mask[1] and x < mask[2] and y < mask[3]:
          return True
    else:
          return False

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

# find false positive via precision recall curve that 
def find_fp(precision: np.ndarray, threshold: np.ndarray):
    idx = np.where(precision < 1.0)
    # idx = precision.where(precision < 1.0)
    return threshold[idx]

# plot precision recall curve
def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, average_precision, method = 'SP+SG', exp_name = 'test'):
    # calculate max f2 score, and max recall
    f_score = 2 * (precision * recall) / (precision + recall)
    f_idx = np.argmax(f_score)
    f_precision, f_recall = np.squeeze(precision[f_idx]), np.squeeze(recall[f_idx])
    r_precision, r_recall = max_recall(precision, recall)
    plt.plot(recall, precision, label="{} (AP={:.5f})".format(method, average_precision))
    plt.vlines(r_recall, np.min(precision), r_precision, colors='green', linestyles='dashed')
    plt.vlines(f_recall, np.min(precision), f_precision, colors='red', linestyles='dashed')
    plt.hlines(f_precision, np.min(recall), f_recall, colors='red', linestyles='dashed')
    plt.hlines(r_precision, np.min(recall), r_recall, colors='green', linestyles='dashed')
    plt.xlim(0, None)
    plt.ylim(np.min(precision), None)
    plt.text(f_recall + 0.005, f_precision + 0.005, '[{:.5f}, {:.5f}]'.format(f_recall, f_precision))
    plt.text(r_recall + 0.005, r_precision + 0.005, '[{:.5f}, {:.5f}]'.format(r_recall, r_precision))
    plt.scatter(f_recall, f_precision, marker='o', color='red', label='Max F score')
    plt.scatter(r_recall, r_precision, marker='o', color='green', label='Max Recall')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curves for {}".format(exp_name))

def plot_pr_curves(recall: list, precision: list, average_precision: list, method = 'test'):
    # plt.figure(figsize=(3.4, 2.5))
    plt.plot(recall, precision, label="{} (AP={:.5f})".format(f'{method}', average_precision))
    r_precision, r_recall = max_recall(precision, recall)
    plt.ylim(np.min(precision), 1.05)
    plt.xlim(0, None)
    plt.vlines(r_recall, np.min(precision), r_precision, colors='green', linestyles='dashed')
    # plt.vlines(r_recall_l, np.min(precision_l), r_precision_l, colors='green', linestyles='dashed')
    plt.scatter(r_recall, r_precision, marker='o', color='green', label='_no_legend_')
    # plt.scatter(r_recall_l, r_precision_l, marker='o', color='green', label='_no_legend_')
    # plt.annotate(f'{r_recall_l:.2f}', (r_recall_l, r_precision), textcoords="offset points", xytext=(5,5), ha='left', fontsize=10)
    plt.annotate(f'{r_recall:.2f}', (r_recall, r_precision), textcoords="offset points", xytext=(5,5), ha='left', fontsize=10)
    plt.legend()
    # plt.savefig('pr-curve.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
    
# perform RANSAC on the matches, 
# default is fundamental matrix
def RANSAC(name0: str, name1: str,
           match_path: Path,
           feature_path: Path, feature_path_r = None, 
           type='F', 
           mask = None
      #      mask = [565, 20, 603, 35]
      ):
      
    
    logger.setLevel('INFO')
#     matches, scores = get_matches(match_path, name0.split('/')[-1], name1.split('/')[-1])
    matches, scores = get_matches(match_path, name0, name1)
    original_matches = len(matches)
    matches = matches[scores > 0.5]
    logger.debug(f'Found {len(matches)} matches between {name0} and {name1}.')
    
    if feature_path_r is None:
            feature_path_r = feature_path
            
#     kpts0 = get_keypoints(feature_path, name0.split('/')[-1])
#     kpts1 = get_keypoints(feature_path_r, name1.split('/')[-1])
    kpts0 = get_keypoints(feature_path, name0)
    kpts1 = get_keypoints(feature_path_r, name1)
    
    logger.debug(f'kp0: {len(kpts0)}, kp1: {len(kpts1)}')
    
    point_map = []
    
    if mask is not None:
        logger.debug(f'Using mask: {mask}')
        
        for match in matches:
            x1, y1, x2, y2 = kpts0[match[0]][0], kpts0[match[0]][1], kpts1[match[1]][0], kpts1[match[1]][1]
            if inmask(x1, y1, mask) and inmask(x2, y2, mask):
                continue
            else:
                point_map.append([x1, y1, x2, y2])
    
    else:
        for match in matches:
            x1, y1, x2, y2 = kpts0[match[0]][0], kpts0[match[0]][1], kpts1[match[1]][0], kpts1[match[1]][1]
            point_map.append([x1, y1, x2, y2])
    
    pointMap = np.array(point_map, dtype=np.float32)
            

    if type == 'F':
        if pointMap.shape[0] < 8:
            logger.debug(f'Not enough points to compute fundamental matrix')
            return pointMap, np.empty((0, 4))
            # return pointMap, pointMap
        
        logger.debug(f'solving Fundamental matrix...\n' + 
                     f'{name0} and {name1}')
        
        F, inliers_idx = cv2.findFundamentalMat(pointMap[:, :2], pointMap[:, 2:], cv2.FM_RANSAC, 3.0)
        
        if F is None or inliers_idx is None:
            logger.debug(f'Failed to compute fundamental matrix')
            return pointMap, np.empty((0, 4))
            # return pointMap, pointMap

        logger.debug(f'F: {F}, inliers_idx: {inliers_idx}')
        
        inliers = pointMap[inliers_idx.ravel() == 1]
        logger.debug(f'original: {len(pointMap)}, inliers: {len(inliers)}')
    
    elif type == 'H':
        if pointMap.shape[0] < 4:
            logger.debug(f'Not enough points to compute homography')
            return pointMap, np.empty((0, 4))
        logger.debug(f'solving homography...\n' + 
                     f'{name0} and {name1}')
        
        H, inliers_idx = cv2.findHomography(pointMap[:, :2], pointMap[:, 2:], cv2.RANSAC, 15.0)

        if H is None or inliers_idx is None:
            logger.debug(f'Failed to compute homography')
            return pointMap, np.empty((0, 4))

        logger.debug(f'H: {H}, inliers_idx: {inliers_idx}')
        inliers = pointMap[inliers_idx.ravel() == 1]
        logger.debug(f'original: {len(pointMap)}, inliers: {len(inliers)}')

    return pointMap, inliers

# evaluation
def eval_from_pair(data: Tuple[str, str, int],
                   match_path: Path,
                   feature_path: Path, feature_path_r = None,
                   fout = None,
                   return_all = False, allow_label = False):
      
      query, reference, label = data
      points_all, inliers = RANSAC(query, reference, match_path, feature_path, feature_path_r, type='F', mask=None, log_stream=fout)

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

def eval_from_pairs(queue: multiprocessing.Queue,
                    data: list,
                    match_path: Path,
                    feature_path: Path, feature_path_r = None):
      
      inliers_list = []
      labels = []
      
      if feature_path_r is None:
            feature_path_r = feature_path
      
      for d in tqdm(data, total=len(data)):
                  inliers, label = eval_from_pair(d, match_path, feature_path, feature_path_r, None, False, True)
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


def Eval_MP(num_process: int,
            dataset: Dataset,
            match_path: Path, 
            feature_path: Path, feature_path_r = None, 
            export_dir = None):
      '''
      TODO: Multiprocessing version of Eval, the acceleration is working now
      
      '''
      
      if feature_path_r is None:
            feature_path_r = feature_path
      
      if export_dir is not None:
            if not Path(export_dir).parent.exists():
                  Path(export_dir).parent.mkdir(parents=True, exist_ok=True)            
            
      logger.info(f'Using {num_process} processes for acceleration.')
      q = multiprocessing.Queue()
      
      num_matches = []
      labels = []
      
      logger.info(f'Evaluating {len(dataset)} pairs')
      
      pairs_list = [dataset[i::num_process] for i in range(num_process)]      
      processes = []      
     
      for i in range(num_process):
            p = multiprocessing.Process(target=eval_from_pairs, args=(q, pairs_list[i], match_path, feature_path, feature_path_r))
            processes.append(p)
            p.start()
      
      for p in processes:
            ret = q.get()
            num_matches.extend(ret[0])
            labels.extend(ret[1])
      
      for p in processes:
            p.join()

      logger.debug(f'num_matches: {num_matches}')
      logger.debug(f'labels: {labels}')
      
      
      matches_pts = np.array(num_matches)
      labels = np.array(labels)
      
      logger.debug(f'matches_pts: {matches_pts}')
      logger.debug(f'Max num of matches is {max(matches_pts)}')
      
      matches_pts_norm = matches_pts / max(matches_pts)
      average_precision = average_precision_score(labels, matches_pts_norm)
      precision, recall, TH = precision_recall_curve(labels, matches_pts_norm)
            
      logger.info(f'Evaluation Done')
      _, r_recall = max_recall(precision, recall)
      logger.info(f'\n' +
            f'Evaluation results: \n' +
            'Average Precision: {:.5f} \n'.format(average_precision) + 
            'Maximum Recall @ 100% Precision: {:.5f} \n'.format(r_recall))
      
      # save the result
      if export_dir is not None:
            np.save(str(export_dir), {'prob': matches_pts_norm, 
                                      'gt': labels, 
                                      'inliers': matches_pts, 
                                      'precision': precision, 
                                      'recall': recall, 
                                      'TH': TH,
                                      'average_precision': average_precision,
                                      'max_recall': r_recall})
      
      
      return precision, recall, average_precision


def eval_from_path_multiprocess(num_process: int,
                                gt_file_path: Path, 
                                match_path: Path, 
                                feature_path: Path, feature_path_r = None, 
                                export_dir = None,
                                return_all = False):
      
      if feature_path_r is None:
            feature_path_r = feature_path
      
      if export_dir is not None:
            if not Path(export_dir).parent.exists():
                  Path(export_dir).parent.mkdir(parents=True, exist_ok=True)            
      
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
     
      for i in range(num_process):
            p = multiprocessing.Process(target=eval_from_pairs, args=(q, pairs_list[i], match_path, feature_path, feature_path_r))
            processes.append(p)
            p.start()
      
      for p in processes:
            ret = q.get()
            num_matches.extend(ret[0])
            labels.extend(ret[1])
      
      for p in processes:
            p.join()

      logger.debug(f'num_matches: {num_matches}')
      logger.debug(f'labels: {labels}')
      
      
      matches_pts = np.array(num_matches)
      labels = np.array(labels)
      
      logger.debug(f'matches_pts: {matches_pts}')
      logger.debug(f'Max num of matches is {max(matches_pts)}')
      
      matches_pts_norm = matches_pts / max(matches_pts)
      average_precision = average_precision_score(labels, matches_pts_norm)
      precision, recall, TH = precision_recall_curve(labels, matches_pts_norm)
      
      if export_dir is not None:
            np.save(str(export_dir), {'prob': matches_pts_norm, 
                                   'gt': labels, 
                                   'inliers': matches_pts, 
                                   'precision': precision, 
                                   'recall': recall, 
                                   'TH': TH, 
                                   'average_precision': average_precision})
      
      if return_all:
            return precision, recall, average_precision, inliers_list, points_all_list
      
      return precision, recall, average_precision, inliers_list


def Eval(dataset: Dataset, matches_path: Path, features: Path, features_ref = None, export_dir = None, seq = False):
      '''
      TODO: Implement Evaluation Function
      
      '''
      
      # Sequence Image
      if seq:
            # pointMap: all the points, inliers: inliers
            pointMaps_all = []
            inliers_list_all = []
            num_matches_all = []
            num_matches_sum = []
            labels = []
            qImages = []
            rImages = []
      
            # Iterate over dataset
            for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
                  # extract data
                  q, r, pairs, label = data
                  pointMaps = []
                  inliers_list = []
                  
                  for pair in pairs:
                        qImage, rImage = pair
                        pointMap, inliers = RANSAC(qImage, rImage, matches_path, features, features_ref)
                        pointMaps.append(pointMap)
                        inliers_list.append(inliers)
                  
                  qImages.append(q)
                  rImages.append(r)
                  num_matches = np.array([len(inliers) for inliers in inliers_list])
                  num_matches_all.append(num_matches)
                  num_matches_sum.append(sum(num_matches))
                  pointMaps_all.append(pointMaps)
                  labels.append(label)
                  inliers_list_all.append(inliers_list)
            
            # Finish Calculation
            logger.info(f'Calculating Matches Done')
            
            num_matches_sum = np.array(num_matches_sum)
            num_matches_norm = num_matches_sum / max(num_matches_sum)
            
            average_precision = average_precision_score(labels, num_matches_norm)
            precision, recall, TH = precision_recall_curve(labels, num_matches_norm)
            
            logger.info(f'Evaluation Done')
            _, r_recall = max_recall(precision, recall)
            
            logger.info(f'\n' +
                  f'Evaluation results: \n' +
                  'Average Precision: {:.5f} \n'.format(average_precision) + 
                  'Maximum Recall @ 100% Precision: {:.5f} \n'.format(r_recall))
            
            # save the result to npy
            if export_dir is not None:
                  logger.info(f'Saving Exp results to {export_dir}')
                  
                  np.save(str(export_dir), {'prob': num_matches_norm,
                                          'qImages': qImages,
                                          'rImages': rImages,
                                          'gt': labels, 
                                          'inliers': inliers_list,
                                          'all_matches': pointMaps,
                                          'precision': precision, 
                                          'recall': recall, 
                                          'TH': TH,
                                          'average_precision': average_precision,
                                          'Max Recall': r_recall})
            
            return precision, recall, average_precision
      
      # Single Image
      else:
            # pointMap: all the points, inliers: inliers
            pointMaps = []
            inliers_list = []
            num_matches = []
            labels = []
            qImages = []
            rImages = []
            
            # Iterate over dataset
            for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
                  # extract data
                  qImage, rImage, label = data
                  
                  # DO RANSAC
                  pointMap, inliers = RANSAC(qImage, rImage, matches_path, features, features_ref)
                  
                  pointMaps.append(pointMap)
                  inliers_list.append(inliers)
                  num_matches.append(len(inliers))
                  labels.append(label)
                  qImages.append(qImage)
                  rImages.append(rImage)
                  

            num_matches = np.array(num_matches)
            num_matches_norm = num_matches / max(num_matches)
            average_precision = average_precision_score(labels, num_matches_norm)
            precision, recall, TH = precision_recall_curve(labels, num_matches_norm)
            
            
            logger.info(f'Evaluation Done')
            _, r_recall = max_recall(precision, recall)
            
            logger.info(f'\n' +
                  f'Evaluation results: \n' +
                  'Average Precision: {:.5f} \n'.format(average_precision) + 
                  'Maximum Recall @ 100% Precision: {:.5f} \n'.format(r_recall))
            
            
            # save the result to npy
            if export_dir is not None:
                  logger.info(f'Saving Exp results to {export_dir}')
                  
                  np.save(str(export_dir), {'prob': num_matches_norm,
                                          'qImages': qImages,
                                          'rImages': rImages,
                                          'gt': labels, 
                                          'inliers': inliers_list,
                                          'all_matches': pointMaps,
                                          'precision': precision, 
                                          'recall': recall, 
                                          'TH': TH,
                                          'average_precision': average_precision,
                                          'Max Recall': r_recall})
            
            return precision, recall, average_precision


def seqmatch(dataset: Dataset, matches_path: Path, features: Path, features_ref = None, output_path = None):
      # Open the file
      if output_path is not None:
            if output_path.exists():
                  logger.warning(f'{output_path} already exists. Jumping matching')
                  return None
            logger.info(f'Writing output to {output_path}')
            if isinstance(output_path, Path):
                  output_path = str(output_path)
            f = open(output_path, 'w')
            f.write('qTimestamp rTimestamp lable num_matches\n')
      
      # data holder
      num_matches_list = []
      labels = []
      
      # Iterate over dataset
      for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
            # extract data
            qTimestamp, rTimestamp, pairs, label = data
            pointMaps = []
            inliers_list = []
            for pair in pairs:
                  qImage, rImage = pair
                  pointMap, inliers = RANSAC(qImage, rImage, matches_path, features, features_ref)
                  pointMaps.append(pointMap)
                  inliers_list.append(inliers)
            num_matches = np.array([len(inliers) for inliers in inliers_list])
            
            logger.debug(f'pointMaps: {pointMaps}')
            logger.info(f'Total Inliers: {sum(num_matches)}')
            
            num_matches_list.append(num_matches)
            labels.append(label)
            if output_path is not None:
                  f.write(f'{qTimestamp} {rTimestamp} {label} {sum(num_matches)}\n')
            
      # Finish Calculation
      logger.info(f'Calculating Matches Done')
      
      # Close the file
      if output_path is not None:
            f.close()
            logger.info(f'Finish writting to {output_path}')
            return num_matches_list, labels
      else:
            return num_matches, labels


def calpr(match_file_path: str, exp_file_path = None):
      # open the file
      if isinstance(match_file_path, Path):
            match_file_path = str(match_file_path)
      logger.info(f'Calculating Precision and Recall from {match_file_path}')
      f = open(match_file_path, 'r')
      
      # skip the first line
      lines = f.readlines()[1:]
      f.close()
      
      # calculate precision and recall
      num_matches = np.array([line.strip('\n').split(' ')[-1] for line in lines], dtype=np.float64)
      labels = np.array([line.strip('\n').split(' ')[-2] for line in lines], dtype=np.int64)
      logger.info(f'Max num of maches: {max(num_matches)}, Min num of matches: {min(num_matches)}')
      
      num_matches_norm = num_matches / max(num_matches)
      average_precision = average_precision_score(labels, num_matches_norm)
      precision, recall, TH = precision_recall_curve(labels, num_matches_norm)
      
      # save the exp result
      if exp_file_path is not None:
            logger.info(f'Saving the result to {exp_file_path}')
            np.save(exp_file_path, {'prob': num_matches_norm, 'gt': labels, 'precision': precision, 'recall': recall, 'TH': TH, 'average_precision': average_precision})
      
      return precision, recall, average_precision
      

            
            
            
            
            
            
            
      
      