from gvbench.viz import plot_matches_from_pair
from gvbench.utils import parse_pairs_from_retrieval

from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
      match_path = Path('dataset/tokyo247/matches/matches-loftr.h5')
      feature_path = Path('dataset/tokyo247/features/loftr-kpts.h5')
      image_path = Path('dataset/tokyo247/images')
      image_pair_path = 'dataset/tokyo247/pairs/pairs_from_retrieval.txt'
      pairs_loader = parse_pairs_from_retrieval(image_pair_path)
      # pairs = [(q,r) for q,r in pairs_loader]
      for q, r in tqdm(pairs_loader):
            plot_matches_from_pair(q, r, match_path, feature_path, image_path, save_dir = 'dataset/tokyo247/plot_matches/loftr')
      
      
      