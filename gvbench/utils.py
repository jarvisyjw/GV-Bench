import cv2
from . import logger
from tqdm import tqdm
import numpy as np


def crop_image(image_dir: str):
    im = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    h, w , c = im.shape
    im = im[:h-160, :, :]
    return im


def load_gt(gt: str):
    querys, references, labels = [], [], []
    logger.info(f'Loading ground truth from {gt}')
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            querys.append(query.split('/')[-1])
            references.append(reference.split('/')[-1])
            labels.append(label)
    return querys, references, labels


def parse_pairs_from_retrieval(pairs: str):
    logger.info(f'Loading pairs from {pairs}')
    
    f = open(pairs, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(' ')
            query, reference = line
            yield query, reference


def parse_pairs(gt: str, allow_label = False):
      
    logger.info(f'Loading ground truth from {gt}')
    logger.debug(f'Allow label: {allow_label}')
    
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            if allow_label:
                  yield query, reference, label
            else:
                  yield query, reference


def write_pairs(file: str, pairs: list):
    logger.info(f'Writing pairs to {file}')
    f = open(file, 'w')
    for pair in tqdm(pairs):
        f.write(f'{pair[0]}, {pair[1]}, {pair[2]}\n')
    logger.info(f'Wrote pairs to {file}. DONE!')
            

def gt_loader(gt: str):
    logger.info(f'Loading ground truth from {gt}')
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            yield set(query, reference, label)
            

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def write_to_pairs(gt: str, pairs: str):
    logger.info(f'Loading ground truth from {gt}')
    loader = parse_pairs(gt)
    gts = [(q,r) for q,r in loader]
    logger.info(f'Writing pairs to {pairs}')
    f = open(pairs, 'w')
    for gt in tqdm(gts):
        f.write(f'{gt[0]} {gt[1]}\n')
        

# transform the matches_pts idx to the pairs idx
def idx2pairs(pairs: list, idx: np.ndarray):
    pairs_idx = []
    for i in idx:
        pair_idx = pairs[i]
        pairs_idx.append(pair_idx)
    
    return np.array(pairs_idx)
        

def export_FP(matches_pts: np.ndarray, pairs, gt: np.ndarray, threshold: float):
    
    if max(matches_pts) > 1:
        matches_pts = matches_pts / max(matches_pts)
    logger.info(f'Using threshold {threshold} to export FP.')
    positive_idx = np.array(np.where(matches_pts > threshold)).flatten().astype(int)
    
    labels = np.array(gt)[positive_idx]
    fp_idx = positive_idx[np.array(np.where(labels == 0)).flatten()]
    
    logger.info(f'Found {len(fp_idx)} FP under threshold = {threshold}.')
    pairs_idx = idx2pairs(pairs, fp_idx)
    logger.debug(f'FP pairs are {pairs_idx}.')
    matches = matches_pts[fp_idx]

    return matches, pairs_idx # return number of matches and FP pairs' idx

def RANSAC(kpts0, kpts1):
    
    point_map = []
    for kp_0, kp_1 in zip(kpts0, kpts1):
        # logger.debug(f'kp_0: {kp_0}, kp_1: {kp_1}')
        x1, y1, x2, y2 = kp_0[0],kp_0[1], kp_1[0], kp_1[1]
        point_map.append([x1, y1, x2, y2])
        
    pointMap = np.array(point_map, dtype=np.float32)
    
    if pointMap.shape[0] < 8:
        logger.warning(f'Not enough points to compute fundamental matrix')
        return pointMap, pointMap

    F, inliers_idx = cv2.findFundamentalMat(pointMap[:, :2], pointMap[:, 2:], cv2.FM_RANSAC, 3.0)
    
    if F is None or inliers_idx is None:
        logger.error(f'Failed to compute fundamental matrix')
        return pointMap, pointMap
    
    logger.debug(f'F: {F}, inliers_idx: {inliers_idx}')
    
    inliers = pointMap[inliers_idx.ravel() == 1]
    logger.debug(f'original: {len(pointMap)}, inliers: {len(inliers)}')

    return pointMap, inliers