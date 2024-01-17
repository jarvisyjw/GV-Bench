import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import pandas as pd
from typing import Dict

from hloc import extract_features, match_features, visualization, match_dense
import multiprocessing, threading
import h5py
from .utils import gt_loader, load_gt, parse_pairs
from . import logger


def merge_h5(feature_paths: list, export_path: str):
      logger.info(f'Merging {feature_paths} to {export_path}')
      with h5py.File('export_path',mode='w') as h5fw:
            for file in feature_paths:
                  h5fr = h5py.File(file,'r') 
                  for obj in h5fr.keys():        
                        h5fr.copy(obj, h5fw)
      logger.info(f'Merged {feature_paths} to {export_path}. DONE!')
      for file in feature_paths:
            os.remove(file)
            

def crop_image(image_dir: str):
    im = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    h, w , c = im.shape
    im = im[:h-160, :, :]
    return im


def extractor(feature: str, image_dir: Path, export_dir: Path):
            
      logger.info(f'Extracting {feature} for {image_dir} images')
      feature_conf = extract_features.confs[feature]
      logger.info(f'Output {feature} to {export_dir}')
      feature_path = export_dir / '{}.h5'.format(feature)
      feature_path = extract_features.main(feature_conf, image_dir, feature_path=feature_path)
      logger.info(f'Extracted {feature} to {feature_path}. DONE!')
      
      return feature_path


def match(export_dir: Path, matcher: str, pairs: Path, feature_path_q: Path, feature_path_r = None):
      pairs_loader = parse_pairs(pairs)
      pairs = [(q, r) for q, r in pairs_loader]
      conf = match_features.confs[matcher]
      if feature_path_r is None:
            feature_path_r = feature_path_q
      logger.info(f'Matching {feature_path_q.name} to {feature_path_r.name} using {matcher}')
      matches = Path(
                export_dir, f'{conf["output"]}.h5')
      match_features.match_from_pairs(conf, pairs, matches, feature_path_q, feature_path_r, True)
      

def crop_images_list(image_dir: Path, export_dir: Path, image_list: list):
      for image_name in tqdm(image_list, total = len(image_list)):
            image_path = image_dir / image_name
            image = crop_image(str(image_path))
            if not cv2.imwrite(str(export_dir / image_path.name), image):
                  raise Exception("Could not write image {}".format(export_dir / image_path.name))
      logger.info(f'Cropped images from {image_dir} to {export_dir}. DONE!')
      

def crop_images_multiprocess(image_dir: str, export_dir: str, num_process: int):
                  
      logger.info(f'Cropping images from {image_dir} and export to {export_dir}')
      image_dir = Path(image_dir)
      export_dir = Path(export_dir)
      if not export_dir.exists():
            export_dir.mkdir(parents=True, exist_ok=True)
      image_list = [image_path.name for image_path in Path(image_dir).glob('**/*.jpg')]
      image_list = [image_list[i::num_process] for i in range(num_process)]
      processes = []
      for i in range(num_process):
            p = multiprocessing.Process(target=crop_images_list, args=(image_dir, export_dir, image_list[i]))
            p.start()
            processes.append(p)
      for p in processes:
            p.join()
      logger.info(f'Cropped images from {image_dir} to {export_dir}. DONE!')


def crop_images(image_dir: str, export_dir: str):
      logger.info(f'Cropping images from {image_dir} and export to {export_dir}')
      image_dir = Path(image_dir)
      export_dir = Path(export_dir)
      if not export_dir.exists():
            export_dir.mkdir(parents=True, exist_ok=True)
      for image_path in tqdm(image_dir.glob('**/*.jpg'), total = len(list(image_dir.glob('**/*.jpg')))):
            # logger.debug(f'Cropping {image_path}')
            image = crop_image(str(image_path))
            if not cv2.imwrite(str(export_dir / image_path.name), image):
                  raise Exception("Could not write image {}".format(export_dir / image_path.name))
      logger.info(f'Cropped images from {image_dir} to {export_dir}. DONE!')
      

def select_crop_images(image_list: str, image_dir: str, export_dir: str, num_process: int):
      logger.info(f'Selecting images from {image_dir}, crop and export to {export_dir}')
      image_dir = Path(image_dir)
      export_dir = Path(export_dir)
      if not export_dir.exists():
            export_dir.mkdir(parents=True, exist_ok=True)      
      
      # image_list = [image_path.name for image_path in Path(image_dir).glob('**/*.jpg')]
      image_list = [image_list[i::num_process] for i in range(num_process)]
      processes = []
      for i in range(num_process):
            p = multiprocessing.Process(target=crop_images_list, args=(image_dir, export_dir, image_list[i]))
            p.start()
            processes.append(p)
      for p in processes:
            p.join()
      logger.info(f'Selected images from {image_dir} to {export_dir}. DONE!')


def loftr(pairs_path: Path, 
          image_dir: Path, 
          match_path: Path, 
          features: Path):
      
      logger.info(f'Matching LoFTR for {pairs_path} images')
      # pairs_loader = parse_pairs(pairs_path)
      # pairs = [(q, r) for q, r in pairs_loader]
      conf = match_dense.confs['loftr']
      
      match_dense.main(conf, 
                       pairs_path, 
                       image_dir, 
                       match_path, features)



def parser():
      parser = argparse.ArgumentParser(description='Extract features from images')
      parser.add_argument('--feature', type=str, default='superpoint', choices=extract_features.confs.keys())
      parser.add_argument('--image_dir', type=str, required=True)
      parser.add_argument('--export_dir', type=str, required=True)
      return parser.parse_args()


if __name__ == '__main__':
      # matcher = 'superpoint+lightglue'
      # pairs = 'dataset/robotcar/gt/robotcar_qAutumn_dbNight.txt'
      # feature_path_q = 'dataset/robotcar/features/Autumn_mini_val/superpoint.h5'
      # feature_path_r = 'dataset/robotcar/features/Night_mini_val/superpoint.h5'
      # export_dir = 'dataset/matches/qAutumn_dbNight/'
      # match(matcher, pairs, Path(feature_path_q), Path(feature_path_r), Path(export_dir))
      # matchers = ['superpoint+lightglue', 'superglue', 'NN-superpoint']
      # args = parser()
      
      '''
      Extract Features
      
            root_dir = Path('dataset/robotcar/')
            features = ['superpoint', 'sift', 'disk']
            for feature in features:
                  extractor(feature, Path(root_dir, 'images'), Path(root_dir, 'features'))
      '''
      root_dir = Path('dataset/robotcar/')
      features = ['superpoint', 'sift', 'disk']
      for feature in features:
            extractor(feature, Path(root_dir, 'images'), Path(root_dir, 'features'))

      '''
      Match Features
      '''
      
      logger.info(f'Matching Superpoint features')
      # matchers = ['superglue', 'NN-superpoint']
      root_dir = Path('dataset/robotcar/gt')
      features_path = Path('dataset/robotcar/features/superpoint.h5')
      pairs_paths = [Path(root_dir, 'robotcar_qAutumn_dbNight.txt'), Path(root_dir, 'robotcar_qAutumn_dbSuncloud.txt')]
      for feature in features:
            if feature == 'superpoint':
                  matchers = ['superglue', 'NN-superpoint']
            if feature == 'sift':
                  matchers = ['NN-ratio']
            if feature == 'disk':
                  matchers = ['disk+lightglue']
            for matcher in matchers:
                  for pairs_path in pairs_paths:
                        output_name = pairs_path.name.split('.')[0]
                        match(Path(root_dir, 'matches', output_name), matcher, pairs_path, features_path)
      
      
      # logger.info(f'Matching SIFT features') # TODO Add SIFT + lightglue
      # matchers = ['NN-ratio']
      # features_path = Path('dataset/robotcar/features/sift.h5')
      # features_path = Path('dataset/robotcar/features/sift.h5')
      # pairs_paths = [Path(root_dir, 'robotcar_qAutumn_dbNight.txt'), Path(root_dir, 'robotcar_qAutumn_dbSuncloud.txt')]
      # processes = []
      # for matcher in matchers:
      #       for pairs_path in pairs_paths:
      #             output_name = pairs_path.name.split('.')[0]
      #             p = multiprocessing.Process(target=match, args=(Path(root_dir, 'matches', output_name), matcher, pairs_path, features_path))
      #             p.start()
      #             processes.append(p)
      
      # for p in processes:
      #       p.join()
                  # match(Path(root_dir, 'matches', output_name), matcher, pairs_path, features_path)
      # root_dir = Path('dataset/robotcar/')
      # features_path = Path('dataset/robotcar/features/loftr_kpts.h5')
      # pairs_paths = [Path(root_dir, 'pairs','qAutumn_dbNight.txt'), Path(root_dir, 'pairs', 'qAutumn_dbSuncloud.txt')]
      # for pairs_path in pairs_paths:
      #       # logger.info(f'Matching LoFTR for {pairs_path} images')
      #       output_match_path = Path('dataset/robotcar/matches', pairs_path.name.split('.')[0], 'loftr.h5')
      #       loftr(pairs_path, Path('dataset/robotcar/images'), output_match_path, features_path)
      
            
      
      # datasets = ['Autumn_mini_val', 'Night_mini_val', 'Suncloud_mini_val']
      # root_dir = 'dataset/'
      # for feature in features:
      #       for dataset in datasets:
      #             image_dir = Path(root_dir, dataset)
      #             export_dir = Path(root_dir, 'features', dataset)
      #             extractor(feature, image_dir, export_dir)

      # pairs = 'dataset/gt/robotcar_qAutumn_dbSuncloud.txt'

      # pairs_loader = parse_pairs(pairs)
      # pairs = [(q, r) for q, r, _ in pairs_loader]
      # print(pairs)
      # feature = 'superpoint'
      # image_dir = Path(root_dir, 'Autumn_val/crop/')
      # export_dir = Path(root_dir, 'Autumn_val/features/')
      # # image_dir = root_dir + 'Suncloud_val/stereo/centre/'
      # # export_dir = root_dir + 'Suncloud_val/crop/'
      # # crop_images_multiprocess(image_dir, export_dir, 10)
      # # extractor_multiprocess(feature, image_dir, export_dir, 10)
      # extractor(feature, image_dir, export_dir)
      # gt = 'dataset/gt/robotcar_qAutumn_dbSuncloud.txt'
      # query, reference, label = load_gt(gt)
      # select_crop_images(query, 'dataset/Autumn_val/stereo/centre/', 'dataset/Autumn_mini_val', 80)