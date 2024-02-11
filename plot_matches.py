from gvbench.viz import plot_matches_from_pair
from gvbench.utils import parse_pairs_from_retrieval, parse_pairs

from pathlib import Path
from tqdm import tqdm


def plot_pair(match_path: Path, feature_path: Path, image_path: Path, query: str, reference: str, label: str, save_dir: Path = None):
      plot_matches_from_pair(query, reference, match_path, feature_path, image_path, save_dir= save_dir, label=label, ransac=True)
      

if __name__ == "__main__":
      
      # input match_path, feature_path, image_path
      match_path = Path('dataset/robotcar/matches/qAutumn_dbRain/matches-loftr_rain_test.h5')
      feature_path = Path('dataset/robotcar/matches/qAutumn_dbRain/feats_matches-loftr.h5')
      image_path = Path('dataset/robotcar/images')
      
      # for a single pair
      # query = 'Autumn_mini_val/1418133195630550.jpg'
      # reference = 'Rain_mini_val/1416907955480651.jpg'
      # label = '1'
      # # , , 1
      # plot_pair(match_path, feature_path, image_path, query, reference, label)
      
      
      # # image_pair_path = 'dataset/tokyo247/pairs/pairs_from_retrieval.txt'
      image_pair_path = 'dataset/robotcar/gt/robotcar_qAutumn_dbRain_test.txt'
      
      pairs_loader = parse_pairs(image_pair_path, True)
      pairs = [(q,r,l) for q,r,l in pairs_loader]
      # # pairs = [(q,r) for q,r in pairs_loader]
      for q, r, l in tqdm(pairs, total = len(pairs)):
            plot_pair(match_path, feature_path, image_path, q, r, l, save_dir= 'dataset/robotcar/viz/matches/qAutumn_dbRain/loftr')
      # # q = 'Autumn_mini_val/1418132448920173.jpg'
      # # r = 'Night_mini_val/1418755603512034.jpg'
      #       plot_matches_from_pair(q, r, match_path, feature_path, image_path, save_dir= 'dataset/robotcar/viz/matches/qAutumn_dbRain/loftr', label=l)