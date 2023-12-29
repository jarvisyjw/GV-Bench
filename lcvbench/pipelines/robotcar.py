import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import os

from hloc import extract_features, match_features, visualization
import multiprocessing, threading
import h5py

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


def parser():
      parser = argparse.ArgumentParser(description='Extract features from images')
      parser.add_argument('--feature', type=str, default='superpoint', choices=extract_features.confs.keys())
      parser.add_argument('--image_dir', type=str, required=True)
      parser.add_argument('--export_dir', type=str, required=True)
      return parser.parse_args()


if __name__ == '__main__':
      # args = parser()
      root_dir = '/mnt/DATA_JW/dataset/LCV_DATASET/robotcar/'
      feature = 'superpoint'
      image_dir = Path(root_dir, 'Autumn_val/crop/')
      export_dir = Path(root_dir, 'Autumn_val/features/')
      # image_dir = root_dir + 'Suncloud_val/stereo/centre/'
      # export_dir = root_dir + 'Suncloud_val/crop/'
      # crop_images_multiprocess(image_dir, export_dir, 10)
      # extractor_multiprocess(feature, image_dir, export_dir, 10)
      extractor(feature, image_dir, export_dir)