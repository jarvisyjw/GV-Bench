from dataloaders.ImagePairDataset import ImagePairDataset
import sys
import argparse
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve
from prettytable import PrettyTable
import numpy as np
from typing import Tuple

### import image-matching-models
sys.path.append('third_party/image-matching-models')
import warnings
warnings.filterwarnings("ignore")
from matching import get_matcher, available_models
from matching.im_models.base_matcher import BaseMatcher
from matching.viz import *
from pathlib import Path
import torch
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', help='Path to the config file')
    parser.add_argument('--support_model', action='store_true', help="Show all image-matching models")
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    
    # Check for config file
    if args.config is None:
        if args.support_model:
            print(f"Available models: {available_models}")
            sys.exit(0)
        else:
            raise ValueError('Please provide a config file')

    # Load the config file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    config = dict2namespace(config)
    
    return config

# wrapper of image-matching-models BaseMatcher's load image
def load_image(path: str | Path, resize: int | Tuple = None, rot_angle: float = 0) -> torch.Tensor:
        return BaseMatcher.load_image(path, resize, rot_angle)

def match(matcher, loader, image_size=512):
    '''
    Args:
        matcher: image-matching-models matcher
        loader: dataloader
        image_size: int, resized shape
        
    Return:
        scores: np.array
    '''
    scores = []
    for data in tqdm(loader, total=len(loader)):
        img0 = load_image(data['img0'], resize=image_size)
        # print(img0.shape)
        img1 = load_image(data['img1'], resize=image_size)
        # print(img1.shape)
        result = matcher(img0, img1)
        num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']
        scores.append(num_inliers)
    # normalize
    scores = np.array(scores)
    scores_norm = (scores - np.min(scores)) / (np.max(scores)- np.min(scores))
    return scores_norm

# max recall @ 100% precision
def max_recall(precision: np.ndarray, recall: np.ndarray):
    idx = np.where(precision == 1.0)
    max_recall = np.max(recall[idx])
    return max_recall

def eval(scores, labels):
    '''
    Args:
        scores: np.array
        labels: np.array
        matcher: name of matcher
        talbe: PrettyTable holder
        
    Return:
        precision: np.array
        recall: np.array
    
    '''
    # mAP
    average_precision = average_precision_score(labels, scores)
    precision, recall, TH = precision_recall_curve(labels, scores)
    # max recall @ 100% precision
    recall_max = max_recall(precision, recall)
    return average_precision, recall_max

def main(config):
    # ransac params, keep it consistent for fairness
    ransac_kwargs = {'ransac_reproj_thresh': 3, 
                     'ransac_conf':0.95, 
                     'ransac_iters':2000} # optional ransac params
    # bench sequence
    gvbench_seq = ImagePairDataset(config.data, transform=None) # load images
    labels = gvbench_seq.label # load labels
    # create result table
    table = PrettyTable()
    table.title = f"GV-Bench:{config.data.name}\n"
    table.field_names = ["Matcher", "mAP", "Max Recall@1.0"]
    # matching loop
    for matcher in config.matcher:
        # create tmp table
        table_tmp = PrettyTable()
        table_tmp.title = f"GV-Bench:{config.data.name}\n"
        table_tmp.field_names = ["Matcher", "mAP", "Max Recall@1.0"]
        assert matcher in available_models, f"Invalid model name. Choose from {available_models}"
        print(f"Runing {matcher}...")
        # load matcher
        if torch.cuda.is_available():
            model = get_matcher(matcher, device='cuda', ransac_kwargs=ransac_kwargs)   
        else:
            raise ValueError('No GPU available')
        # compute scores
        scores = match(model, gvbench_seq, image_size=(config.data.image_height, config.data.image_width))
        mAP, MaxR = eval(scores, labels)
        table_tmp.add_row([matcher, mAP, MaxR])
        print(table_tmp)   
        table.add_rows([[matcher, mAP, MaxR]])
    # print result
    print(table)
    
if __name__ == "__main__":
    # parser
    cfg = parser()
    main(cfg)