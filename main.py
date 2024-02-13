import argparse
from pathlib import Path
from tqdm import tqdm

# import sys
# sys.path.append('third_party/Hierarchical-Localization/')

from hloc import extract_features, match_features, visualization, match_dense

from gvbench import logger
from gvbench.utils import *
from gvbench.evaluation import *
from gvbench.viz import *


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
                export_dir, f'{conf["output"]}.h5')
      match_features.match_from_pairs(conf, pairs, matches, feature_path_q, feature_path_r, False)


# def crop_images(image_dir: str, export_dir: str):
      
#       logger.info(f'Cropping images from {image_dir} and export to {export_dir}')
#       image_dir = Path(image_dir)
#       export_dir = Path(export_dir)
#       if not export_dir.exists():
#             export_dir.mkdir(parents=True, exist_ok=True)
#       for image_path in tqdm(image_dir.glob('**/*.jpg'), total = len(list(image_dir.glob('**/*.jpg')))):
#             # logger.debug(f'Cropping {image_path}')
#             image = io.crop_image(str(image_path))
#             if not cv2.imwrite(str(export_dir / image_path.name), image):
#                   raise Exception("Could not write image {}".format(export_dir / image_path.name))
#       logger.info(f'Cropped images from {image_dir} to {export_dir}. DONE!')


def parser():
      parser = argparse.ArgumentParser(description='Extract features from images')
      parser.add_argument('--feature', type=str, default='superpoint', choices=extract_features.confs.keys())
      parser.add_argument('--image_dir', type=str, required=True)
      parser.add_argument('--export_dir', type=str, required=True)
      return parser.parse_args()


if __name__ == '__main__':
      
      
      
      
      # features = ['superpoint_max', 'sift', 'disk']
      # features = ['cosplace']

      # features_path = Path('dataset/Nordland/features')
      # image_path = Path(f'dataset/Nordland/images/')
      # extractor('cosplace', image_path, features_path)
      # export_path = Path(f'dataset/Nordland/features')
      
      # pairs_path = Path('dataset/Nordland/pairs_netvlad.txt')

      # for feature in features:
      #       extractor(feature, image_path, export_path)
      
      # root_dir = Path('dataset/Nordland/')
      pairs_path = Path('dataset/robotcar/pairs/rain_test.txt')
      feature_path = Path('dataset/robotcar/features/qAutumn_dbRain/sift.h5')
      export_dir = Path('dataset/robotcar/matches/qAutumn_dbRain/matches-sift-nn.h5')
      match_features.main(match_features.confs['NN-ratio'], pairs_path, feature_path, matches=export_dir)
      
      # features_path = Path('dataset/mapillary_sls/train_val/boston/features/')
      # pairs_paths = [Path('dataset/mapillary_sls/pairs/boston_cosplace.txt'), Path('dataset/mapillary_sls/pairs/boston.txt')]
      # match(Path(root_dir, 'matches', 'qAutumn_dbNight', 'sp_max'), 'superglue', pairs_path,  Path(root_dir, 'features', 'superpoint_max' + '.h5'))
      # for feature in features:
      #       if feature == 'superpoint_max':
      #             matchers = ['superglue', 'NN-superpoint']
      #       if feature == 'sift':
      #             matchers = ['NN-ratio']
      #       if feature == 'disk':
      #             matchers = ['disk+lightglue', 'NN-ratio']
      #       for matcher in matchers:
      #             match(Path(root_dir, 'matches'), matcher, pairs_path,  Path(root_dir, 'features', feature + '.h5'))
      
      
      # image_path = Path('dataset/mapillary_sls/train_val/boston')
      # features = ['cosplace']
      # export_path = Path('dataset/mapillary_sls/train_val/boston/features')
      # for feature in features:
      #       extractor(feature, image_path, export_path)
      
      # feature = 'orb'
      # feature = 'cosplace'
      # extractor(feature, Path('dataset/robotcar/images/'), Path('dataset/robotcar/features/'))
      # conf = match_dense.confs['loftr']
      # image_path = Path('dataset/robotcar/images')
      
      # pairs_path = Path(f'dataset/robotcar/pairs/qAutumn_dbRain.txt')
      # matches_path = Path(f'dataset/robotcar/matches/robotcar_qAutumn_dbRain/matches-loftr.h5')
      # feature_path = Path('dataset/robotcar/matches/robotcar_qAutumn_dbRain/kpts-loftr.h5')
      # logger.info(f'Matching Robotcar Rain dataset \n' + 
      #             f'pairs_path: {pairs_path} \n'  +
      #             f'matches_path: {matches_path} \n' +
      #                   f'feature_path: {feature_path}')

      # if not feature_path.exists():
      #       feature_path.parent.mkdir(parents=True, exist_ok=True)
      
      # if not matches_path.exists():
      #       matches_path.parent.mkdir(parents=True, exist_ok=True)

      # match_dense.match_and_assign(conf, pairs_path, image_path, matches_path, feature_path)
      
      # image_path = Path('dataset/robotcar/images')
      # feature = 'superpoint_max'
      # extractor(feature, image_path, Path('dataset/robotcar/features/'))
      # feature = 'cosplace'
      # # features = ['sift', 'superpoint', 'disk', 'netvlad']
      # export_path = Path('dataset/tokyo247/features')
      # # for feature in features:
      # extractor(feature, image_path, export_path)
      
      # conf = match_dense.confs['loftr']
      
      
      # pairs_path = Path(f'dataset/robotcar/pairs/qAutumn_dbRain.txt')
      # matches_path = Path(f'dataset/robotcar/matches/robotcar_qAutumn_dbRain/matches-loftr.h5')
      # feature_path = Path('dataset/robotcar/features/qAutumn_dbRain/kpts-loftr.h5')
      # logger.info(f'Matching robotcar dataset \n' + 
      #             f'pairs_path: {pairs_path} \n'  +
      #             f'matches_path: {matches_path} \n' +
      #                   f'feature_path: {feature_path}')

      # if not feature_path.exists():
      #       feature_path.parent.mkdir(parents=True, exist_ok=True)
      
      # if not matches_path.exists():
      #       matches_path.parent.mkdir(parents=True, exist_ok=True)

      # match_dense.match_and_assign(conf, pairs_path, image_path, matches_path, feature_path)
      
      # image_path = Path('dataset/mapillary_sls/train_val/boston')
      # features = ['superpoint']
      
      # root_dir = Path('dataset/mapillary_sls/train_val/boston/')
      # # features_path = Path('dataset/mapillary_sls/train_val/boston/features/')
      # pairs_paths = [Path('dataset/mapillary_sls/pairs/boston_cosplace.txt'), Path('dataset/mapillary_sls/pairs/boston.txt')]
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
      
      
      # conf = match_dense.confs['loftr']
      
      # for dataset in ['qAutumn_dbSnow']:
      #       pairs_path = Path(f'dataset/robotcar/pairs/{dataset}.txt/')
      #       matches_path = Path(f'dataset/robotcar/matches/robotcar_{dataset}/matches-loftr.h5')
      #       feature_path = Path('dataset/robotcar/features/kpts-loftr.h5')
      #       logger.info(f'Matching {dataset} dataset \n' + 
      #                   f'pairs_path: {pairs_path} \n'  +
      #                   f'matches_path: {matches_path} \n ' +
      #                         f'feature_path: {feature_path}')

      #       if not feature_path.exists():
      #             feature_path.parent.mkdir(parents=True, exist_ok=True)
            
      #       if not matches_path.exists():
      #             matches_path.parent.mkdir(parents=True, exist_ok=True)

      #       match_dense.match_and_assign(conf, pairs_path, image_path, matches_path, feature_path)

      # conf = match_dense.confs['loftr']
      # image_path = Path('dataset/robotcar/images')

      
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
      
      # gt_file_path = f'dataset/robotcar/gt/robotcar_qAutumn_dbRain.txt'
      # match_path = Path('dataset/robotcar/matches/robotcar_qAutumn_dbRain/matches-superpoint-superglue.h5')
      # feature_path = Path('dataset/robotcar/features/qAutumn_dbRain/superpoint.h5')
      # scores, labels = score_from_path(gt_file_path, match_path, feature_path)
      # average_precision = average_precision_score(labels, scores)
      # precision, recall, TH = precision_recall_curve(labels, scores)
      # export_FP(gt_file_path, match_path, feature_path, TH)
      


      # precision, recall, average_precision, inliers_list = eval_from_path_multiprocess(20, gt_file_path, match_path, feature_path)
      # plot_pr_curve(recall, precision, average_precision, 'Suncloud2Rain', 'LoFTR')
      # _, r_recall = max_recall(precision, recall)

      # logger.info(f'\n' +
      #             f'Evaluation results: \n' +
      #             'Average Precision: {:.3f} \n'.format(average_precision) + 
      #             'Maximum Recall @ 100% Precision: {:.3f} \n'.format(r_recall))
      # output_path = Path(f'dataset/robotcar/exps/qAutumn_dbNight/LoFTR/pr_curve.png')
      # if not output_path.parent.exists():
      #             output_path.parent.mkdir(parents=True, exist_ok=True)
      # plt.savefig(str(output_path))