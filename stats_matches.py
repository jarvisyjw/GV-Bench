from pathlib import Path
from gvbench.utils import parse_pairs
from gvbench.evaluation import RANSAC


if __name__ == "__main__":
      

      match_path = Path('dataset/robotcar/matches/qAutumn_dbRain/matches-loftr_rain_test.h5')
      feature_path = Path('dataset/robotcar/matches/qAutumn_dbRain/feats_matches-loftr.h5')
      image_pair_path = 'dataset/robotcar/gt/robotcar_qAutumn_dbRain_test.txt'
      
      pairs_loader = parse_pairs(image_pair_path, True)
      pairs = [(q,r,l) for q,r,l in pairs_loader]
      stats = []
      
      for q, r, l in pairs:
            pointMap, inliers = RANSAC(q, r, match_path, feature_path)
            stats.append([q, r, l, len(pointMap), len(inliers)])
      
      print(stats)