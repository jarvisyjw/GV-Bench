from gvbench.viz import plot_matches_from_pair
from gvbench.utils import parse_pairs_from_retrieval, parse_pairs

from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
      match_path = Path('dataset/uacampus/matches/matches-disk-lightglue.h5')
      feature_path = Path('dataset/Nordland_RAS2020/matches/loftr/feats_matches-loftr.h5')
      image_path = Path('dataset/Nordland_RAS2020/images')
      # image_pair_path = 'dataset/uacampus/pairs_netvlad.txt'
      gt_path = 'dataset/Nordland_RAS2020/netvlad_pairs_gt2.txt'
      pairs_loader = parse_pairs(gt_path, True)
      # pairs_loader = parse_pairs_from_retrieval(image_pair_path)
      # pairs = [(q,r,label) for q,r in pairs_loader]
      for q, r, l in tqdm(pairs_loader):
            plot_matches_from_pair(q, r, match_path, feature_path, image_path, save_dir = 'dataset/Nordland_RAS2020/viz/loftr_matches', label = l)
      
      
      