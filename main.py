import argparse
from pathlib import Path
from tqdm import tqdm

from hloc import extract_features, match_features, visualization, match_dense

from gvbench import logger


def extractor(feature: str, image_dir: str, export_dir: str):
      
      logger.info(f'Extracting {feature} for {image_dir} images')
      feature_conf = extract_features.confs[feature]
      logger.info(f'Output {feature} to {export_dir}')
      feature_path = export_dir / '{}.h5'.format(feature)
      feature_path = extract_features.main(feature_conf, image_dir, feature_path=feature_path)
      logger.info(f'Extracted {feature} to {feature_path}. DONE!')
      
      return feature_path


def parser():
      parser = argparse.ArgumentParser(description='Extract features from images')
      parser.add_argument('--feature', type=str, default='superpoint', choices=extract_features.confs.keys())
      parser.add_argument('--image_dir', type=str, required=True)
      parser.add_argument('--export_dir', type=str, required=True)
      return parser.parse_args()


if __name__ == '__main__':
      # # args = parser()
      # root_dir = '/mnt/DATA_JW/dataset/LCV_DATASET/robotcar/'
      # image_dir = root_dir + 'Autumn_val/stereo/centre/'
      # export_dir = root_dir + 'Autumn_val/crop/'
      # crop_images(image_dir, export_dir)
      # # extractor(args.feature, image_dir, export_dir)
      
      
      # image_path = Path('dataset/tokyo247/images')
      # features = ['sift', 'superpoint', 'disk', 'netvlad']
      # export_path = Path('dataset/tokyo247/features')
      # for feature in features:
      #       extractor(feature, image_path, export_path)
      # image_path = Path('dataset/robotcar/images')
      # features = ['sift', 'superpoint', 'disk', 'netvlad']
      # export_path = Path('dataset/robotcar/features')
      # for feature in features:
      #       extractor(feature, image_path, export_path)
      
      # conf = match_dense.confs['loftr']
      
      
      # pairs_path = Path(f'dataset/tokyo247/pairs/pairs_from_retrieval.txt/')
      # matches_path = Path(f'dataset/robotcar/matches/matches-loftr.h5')
      # feature_path = Path('dataset/robotcar/features/kpts-loftr.h5')
      # logger.info(f'Matching Tokyo247 dataset \n' + 
      #             f'pairs_path: {pairs_path} \n'  +
      #             f'matches_path: {matches_path} \n' +
      #                   f'feature_path: {feature_path}')

      # if not feature_path.exists():
      #       feature_path.parent.mkdir(parents=True, exist_ok=True)
      
      # if not matches_path.exists():
      #       matches_path.parent.mkdir(parents=True, exist_ok=True)

      # match_dense.match_and_assign(conf, pairs_path, image_path, matches_path, feature_path)
      
      
      
      # conf = match_dense.confs['loftr']
      
      # for dataset in ['qAutumn_dbSnow']:
      #       pairs_path = Path(f'dataset/robotcar/pairs/{dataset}.txt/')
      #       matches_path = Path(f'dataset/robotcar/matches/robotcar_{dataset}/matches-loftr.h5')
      #       feature_path = Path('dataset/robotcar/features/kpts-loftr.h5')
      #       logger.info(f'Matching {dataset} dataset \n' + 
      #                   f'pairs_path: {pairs_path} \n'  +
      #                   f'matches_path: {matches_path} \n' +
      #                         f'feature_path: {feature_path}')

      #       if not feature_path.exists():
      #             feature_path.parent.mkdir(parents=True, exist_ok=True)
            
      #       if not matches_path.exists():
      #             matches_path.parent.mkdir(parents=True, exist_ok=True)

      #       match_dense.match_and_assign(conf, pairs_path, image_path, matches_path, feature_path)
