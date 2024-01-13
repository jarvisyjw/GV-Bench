import argparse
from pathlib import Path
from tqdm import tqdm
import cv2

# import sys
# sys.path.append('third_party/Hierarchical-Localization/')

from hloc import extract_features, match_features, visualization

from gvbench import logger
from gvbench.utils import crop


def extractor(feature: str, image_dir: str, export_dir: str):
      
      logger.info(f'Extracting {feature} for {image_dir} images')
      feature_conf = extract_features.confs[feature]
      logger.info(f'Output {feature} to {export_dir}')
      feature_path = export_dir / '{}.h5'.format(feature)
      feature_path = extract_features.main(feature_conf, image_dir, feature_path=feature_path)
      logger.info(f'Extracted {feature} to {feature_path}. DONE!')
      
      return feature_path


def crop_images(image_dir: str, export_dir: str):
      
      logger.info(f'Cropping images from {image_dir} and export to {export_dir}')
      image_dir = Path(image_dir)
      export_dir = Path(export_dir)
      if not export_dir.exists():
            export_dir.mkdir(parents=True, exist_ok=True)
      for image_path in tqdm(image_dir.glob('**/*.jpg'), total = len(list(image_dir.glob('**/*.jpg')))):
            # logger.debug(f'Cropping {image_path}')
            image = crop.crop_image(str(image_path))
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
      image_dir = root_dir + 'Autumn_val/stereo/centre/'
      export_dir = root_dir + 'Autumn_val/crop/'
      crop_images(image_dir, export_dir)
      # extractor(args.feature, image_dir, export_dir)
