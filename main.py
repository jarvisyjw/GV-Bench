import argparse
from pathlib import Path
from tqdm import tqdm

from hloc import extract_features, match_features, visualization, match_dense

from gvbench import logger
from gvbench.utils import parse_pairs_from_retrieval


def extractor(feature: str, image_dir: Path, export_dir: Path):
      
      logger.info(f'Extracting {feature} for {image_dir} images')
      feature_conf = extract_features.confs[feature]
      logger.info(f'Output {feature} to {export_dir}')
      feature_path = export_dir / '{}.h5'.format(feature)
      feature_path = extract_features.main(feature_conf, image_dir, feature_path=feature_path)
      logger.info(f'Extracted {feature} to {feature_path}. DONE!')
      
      return feature_path

def match(export_dir: Path, matcher: str, pairs: Path, feature_path_q: Path, feature_path_r = None):
      pairs_loader = parse_pairs_from_retrieval(pairs)
      pairs = [(q, r) for q, r in pairs_loader]
      conf = match_features.confs[matcher]
      if feature_path_r is None:
            feature_path_r = feature_path_q
      logger.info(f'Matching {feature_path_q.name} to {feature_path_r.name} using {matcher}')
      matches = Path(
                export_dir, feature_path_q.stem, f'{conf["output"]}.h5')
      match_features.match_from_pairs(conf, pairs, matches, feature_path_q, feature_path_r, False)

def parser():
      parser = argparse.ArgumentParser(description='Extract features from images')
      parser.add_argument('--feature', type=str, default='superpoint', choices=extract_features.confs.keys())
      parser.add_argument('--image_dir', type=str, required=True)
      parser.add_argument('--export_dir', type=str, required=True)
      return parser.parse_args()


if __name__ == '__main__':
      
      # args = parser()
      # features = ['disk', 'superpoint_max', 'sift']
      # image_dir = Path('dataset/robotcar/images')
      # export_dir = Path('dataset/robotcar/features')
      # image_dir = Path('dataset/Nordland_RAS2020/images')
      # export_dir = Path('dataset/Nordland_RAS2020/features')
      # for feature in features:
      #       extractor(feature, image_dir, export_dir)
      # extractor(args.feature, Path(args.image_dir), Path(args.export_dir))
      
      # image_dir = 
      # feature = ['cosplace']
      # extractor(feature, image_dir, export_dir)
      
      # features = ['disk']
      # features_path = Path('dataset/robotcar/features')
      # pairs_path = Path('dataset/robotcar/pairs/qAutumn_dbSuncloud.txt')
      # root_dir = Path('dataset/robotcar/')
      pairs_path = Path('dataset/uacampus/cosplace_10.txt')
      feature_path = Path('dataset/uacampus/features/disk.h5')
      export_dir = Path('dataset/uacampus/matches_new/disk-lightglue.h5')
      match_features.main(match_features.confs['disk+lightglue'], pairs_path, feature_path, matches=export_dir)

      # for feature in features:
      #       if feature == 'superpoint_max':
      #             matchers = ['superglue', 'NN-superpoint']
      #       if feature == 'sift':
      #             matchers = ['NN-ratio']
      #       if feature == 'disk':
      #             matchers = ['NN-ratio']
      #       for matcher in matchers:
      #             match(Path(root_dir, 'matches', 'qAutumn_dbSuncloud'), matcher, pairs_path,  Path(root_dir, 'features', feature + '.h5'))

      
      # features = ['disk', 'superpoint_max', 'sift']
      # features_path = Path('dataset/Nordland_RAS2020/features')
      # pairs_path = Path('dataset/Nordland_RAS2020/cosplace_pairs.txt')
      # root_dir = Path('dataset/Nordland_RAS2020/')

      # for feature in features:
      #       if feature == 'superpoint_max':
      #             matchers = ['superglue', 'NN-superpoint']
      #       if feature == 'sift':
      #             matchers = ['NN-ratio']
      #       if feature == 'disk':
      #             matchers = ['disk+lightglue', 'NN-ratio']
      #       for matcher in matchers:
      #             match(Path(root_dir, 'matches'), matcher, pairs_path,  Path(root_dir, 'features', feature + '.h5'))

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
      # image_path = Path('dataset/tokyo247/images')
      # feature = 'cosplace'
      # # features = ['sift', 'superpoint', 'disk', 'netvlad']
      # export_path = Path('dataset/tokyo247/features')
      # # for feature in features:
      # extractor(feature, image_path, export_path)
      
      # conf = match_dense.confs['loftr']
      
      
      # pairs_path = Path(f'dataset/tokyo247/pairs/cosplace.txt/')
      # matches_path = Path(f'dataset/tokyo247/matches/matches-loftr.h5')
      # feature_path = Path('dataset/tokyo247/features/kpts-loftr.h5')
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
      
      # for dataset in ['qAutumn_dbRain']:
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
      
      # features = ['superpoint']
      
      # root_dir = Path('dataset/mapillary_sls/train_val/boston/')
      # # # features_path = Path('dataset/mapillary_sls/train_val/boston/features/')
      # pairs_path = Path('dataset/mapillary_sls/pairs/boston.txt')
      # match(Path(root_dir, 'matches', 'boston'), 'superglue', pairs_path,  Path(root_dir, 'features', 'superpoint' + '.h5'))

      
      # for feature in features:
      #       if feature == 'superpoint':
      #             matchers = ['superglue']
      #       if feature == 'sift':
      #             matchers = ['NN-ratio']
      #       if feature == 'disk':
      #             matchers = ['disk+lightglue']
      #       for matcher in matchers:
      #             for pairs_path in pairs_paths:
      #                   output_name = pairs_path.name.split('.')[0]
      #                   match(Path(root_dir, 'matches', output_name), matcher, pairs_path,  Path(root_dir, 'features', feature + '.h5'))
      