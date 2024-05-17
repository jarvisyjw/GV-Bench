from hloc import extract_features, match_features
from pathlib import Path
from argparse import ArgumentParser

def parser():
      parser = ArgumentParser()
      parser.add_argument('--image_path', type=str)
      parser.add_argument('--output_path', type=str)
      parser.add_argument('--feature', type=str)
      parser.add_argument('--featues', type=Path)
      parser.add_argument('--pairs', type=Path)
      parser.add_argument('--features_ref', type=Path)
      
      return parser.parse_args()


if __name__ == "__main__":
      args = parser()
      ### Extract features of each sequences
      # conf = extract_features.confs[args.feature]
      # extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path))
      
      ### Match feature of sequences
      match_features.main(match_features.confs['superglue'], pairs = Path(args.pairs), features= Path(args.features), features_ref= Path(args.features_ref), matches= Path(args.output_path))
      