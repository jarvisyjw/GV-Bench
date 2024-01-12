import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import pandas as pd

from hloc import extract_features, match_features, visualization, match_features
import multiprocessing, threading
import h5py
from ..utils.io import gt_loader, load_gt, parse_pairs

from .. import logger


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


def match(matcher: str, pairs: Path, feature_path_q: Path, feature_path_r: Path, export_dir: Path):
      pairs_loader = parse_pairs(pairs)
      pairs = [(q, r) for q, r, _ in pairs_loader]
      conf = match_features.confs[matcher]
      logger.info(f'Matching {feature_path_q.name} to {feature_path_r.name} using {matcher}')
      matches = Path(
                export_dir, f'{conf["output"]}.h5')
      match_features.match_from_pairs(conf, pairs, matches, feature_path_q, feature_path_r)
      

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


def find_closest(lst, x):
    """
    在有序列表 lst 中找到与给定值 x 大小最接近的元素。
    """
    n = len(lst)
    if x <= lst[0]:
        return lst[0]
    elif x >= lst[n-1]:
        return lst[n-1]
    else:
        # 二分查找
        low, high = 0, n-1
        while low <= high:
            mid = (low + high) // 2
            if lst[mid] == x:
                return lst[mid]
            elif lst[mid] < x:
                low = mid + 1
            else:
                high = mid - 1

        # 最后返回距离 x 最近的元素
        if lst[high] - x < x - lst[low]:
            return lst[high]
        else:
            return lst[low]


# def cal_two_frame_dis_yaw(query_dir, ref_dir, query_t, ref_t):
#     query_ins_fname = Path(query_dir, "gps/ins.csv")
#     query_df = pd.read_csv(query_ins_fname)
#     query_df_timestamps = query_df["timestamp"].tolist()
#     ref_ins_fname = Path(ref_dir, "gps/ins.csv")
#     ref_df = pd.read_csv(ref_ins_fname)
#     ref_df_timestamps = ref_df["timestamp"].tolist()

#     closest_q_mini_t = find_closest(query_df_timestamps, query_t)
#     q_north, q_east, q_yaw = query_df[query_df["timestamp"]==closest_q_mini_t]["northing"].item(), query_df[query_df["timestamp"]==closest_q_mini_t]["easting"].item(), query_df[query_df["timestamp"]==closest_q_mini_t]["yaw"].item()
#     # print(q_north, q_east)
#     closest_ref_mini_t = find_closest(ref_df_timestamps, ref_t)
#     ref_north, ref_east, ref_yaw = ref_df[ref_df["timestamp"]==closest_ref_mini_t]["northing"].item(), ref_df[ref_df["timestamp"]==closest_ref_mini_t]["easting"].item(), ref_df[ref_df["timestamp"]==closest_ref_mini_t]["yaw"].item()
#     dis = calculate_distance(q_north, q_east, ref_north, ref_east)
#     view = calculate_view(q_yaw, ref_yaw)
#     dis = calculate_distance(q_north, q_east, ref_north, ref_east)
#     print("Distance:", dis)
#     print("View:", view)


def parser():
      parser = argparse.ArgumentParser(description='Extract features from images')
      parser.add_argument('--feature', type=str, default='superpoint', choices=extract_features.confs.keys())
      parser.add_argument('--image_dir', type=str, required=True)
      parser.add_argument('--export_dir', type=str, required=True)
      return parser.parse_args()


if __name__ == '__main__':
      matcher = 'superpoint+lightglue'
      pairs = 'dataset/gt/robotcar_qAutumn_dbSuncloud.txt'
      feature_path_q = 'dataset/features/Autumn_mini_val/superpoint.h5'
      feature_path_r = 'dataset/features/Suncloud_mini_val/superpoint.h5'
      export_dir = 'dataset/matches/qAutumn_dbSuncloud/'
      match(matcher, pairs, Path(feature_path_q), Path(feature_path_r), Path(export_dir))
      # matchers = ['superpoint+lightglue', 'superglue', 'NN-superpoint']
      # args = parser()
      # root_dir = '/mnt/DATA_JW/dataset/LCV_DATASET/robotcar/'
      # features = ['superpoint', 'sift', 'disk']
      # feature = 'sift'
      # image_dir = Path('dataset/')
      # export_dir = Path('features')
      # extractor(feature, image_dir, export_dir)
      
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