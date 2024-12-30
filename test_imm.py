from dataloaders.ImagePairDataset import ImagePairDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import sys
import argparse
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve
import h5py
from prettytable import PrettyTable

### import image-matching-models
sys.path.append('third_party/image-matching-models')
from matching import get_matcher, available_models
from matching.viz import *
from pathlib import Path
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file')
    
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
    
    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    return config

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
    for image0, image1, label in tqdm(loader, total=len(loader)):
        img0 = matcher.load_image(image0, resize=image_size)
        img1 = matcher.load_image(image1, resize=image_size)
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
    
    Return:
        precision: np.array
        recall: np.array
    
    '''
    # mAP
    average_precision = average_precision_score(labels, scores)
    precision, recall, TH = precision_recall_curve(labels, scores)
    recall_max = max_recall(precision, recall)
    
    table = PrettyTable()
    table.field_names = ["mAP", "Max Recall@1.0"]
    table.add_rows(
    [
        [average_precision, recall_max]
    ])
    
    print(table)
    
    return precision, recall

def main(config):
    # Check if CUDA is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("CUDA is not available.")
    
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    ransac_kwargs = {'ransac_reproj_thresh': 3, 
                        'ransac_conf':0.95, 
                        'ransac_iters':2000} # optional ransac params
    
    if torch.cuda.is_available():
        matcher = get_matcher([matcher], device='cpu', **ransac_kwargs) #try an ensemble!
        matcher = torch.nn.DataParallel(matcher).cuda()
    else:
        matcher = get_matcher([matcher], device='cpu:0', **ransac_kwargs) #try an ensemble!
    
    gvbench_seq = ImagePairDataset(config.data, transform=None)
    loader = DataLoader(gvbench_seq, batch_size=1, shuffle=False)
    scores = match(matcher, loader, image_size=(config.data.image_height, config.data.image_width))
    labels = gvbench_seq.label
    precision, recall = eval(scores, labels)

if __name__ == "__main__":
    # parser
    cfg = parser()
    main(cfg)
    
    # prepare dataset
    # gvbench_seq = ImagePairDataset(config.data, transform=None)
    # print(f"Num of pairs: {len(gvbench_seq)}")
    # init image matcher
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Using device: {device}')
    # w = config.data.image_width
    # h = config.data.image_height
    
    # for matcher in config.matcher:
    #     print(f'Using matcher: {matcher}')
    #     assert matcher in available_models, f'{matcher} is not available'
    #     # matcher = config.matcher
    #     ransac_kwargs = {'ransac_reproj_thresh': 3, 
    #                     'ransac_conf':0.95, 
    #                     'ransac_iters':2000} # optional ransac params
    #     matcher = get_matcher([matcher], device=device, **ransac_kwargs) #try an ensemble!
    #     # resize images to 512
    #     image_size = (h,w)
    #     # Perform Matching
    #     scores = match(matcher, gvbench_seq, image_size=image_size)
    #     labels = gvbench_seq.label
    #     # Evaluation
    #     precision, recall = eval(scores, labels)