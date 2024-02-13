from pathlib import Path
import pandas as pd
from tqdm import tqdm
from gvbench.utils import parse_pairs
from gvbench.evaluation import RANSAC, get_keypoints
from gvbench import logger


if __name__ == "__main__":
      
      logger.setLevel('INFO')
      match_path = Path('dataset/robotcar/matches/qAutumn_dbRain/matches-loftr_rain_test.h5')
      feature_path = Path('dataset/robotcar/matches/qAutumn_dbRain/feats_matches-loftr.h5')
      image_pair_path = 'dataset/robotcar/gt/robotcar_qAutumn_dbRain_test.txt'
      
      pairs_loader = parse_pairs(image_pair_path, True)
      pairs = [(q,r,l) for q,r,l in pairs_loader]
      stats = []
      
      for q, r, l in tqdm(pairs):
            pointMap, inliers = RANSAC(q, r, match_path, feature_path)
            kpts0 = get_keypoints(feature_path, q)
            kpts1 = get_keypoints(feature_path, r)
            stats.append([q, r, l, len(pointMap), len(inliers), len(kpts0), len(kpts1)])
      
      df = pd.DataFrame(stats, columns=['query', 'reference', 'label', 'matches', 'inliers', 'kpts0', 'kpts1'])
      
      # print(df)
      
      df.to_csv('dataset/robotcar/exps/qAutumn_dbRain/loftr_stats.csv', index=True)