from hloc import extract_features, match_features, match_dense
from pathlib import Path
from argparse import ArgumentParser

def parser():
      parser = ArgumentParser()
      parser.add_argument('--image_path', type=str)
      parser.add_argument('--output_path', type=str)
      parser.add_argument('--feature', type=str)
      parser.add_argument('--features', type=Path)
      parser.add_argument('--pairs', type=Path)
      parser.add_argument('--features_ref', type=Path)
      parser.add_argument('--matching', action='store_true')
      parser.add_argument('--extraction', action='store_true')
      
      return parser.parse_args()


if __name__ == "__main__":
      args = parser()
      if args.extraction:
            ### Extract features of each sequences
            features = ['superpoint_max', 'sift', 'disk']
            for featue in features:
                  conf = extract_features.confs[args.feature]
                  extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path, f'{featue}.h5'))
      
      if args.match:
            ### Match feature of sequences
            features = ['superpoint_max', 'sift', 'disk']
            for feature in features:
                  if feature == 'superpoint_max':
                        match_features.main(match_features.confs['superglue'], pairs = Path(args.pairs), features= Path(args.output_path), matches= Path(args.output_path))
                        match_features.main(match_features.confs['NN-superpoint'], pairs = Path(args.pairs), features= Path(args.output_path), matches= Path(args.output_path))
                  if feature == 'disk':
                        match_features.main(match_features.confs['disk+lightglue'], pairs = Path(args.pairs), features= Path(args.output_path), matches= Path(args.output_path))
                        match_features.main(match_features.confs['NN-ratio'], pairs = Path(args.pairs), features= Path(args.output_path), matches= Path(args.output_path))
                  if feature == 'sift':
                        match_features.main(match_features.confs['NN-ratio'], pairs = Path(args.pairs), features= Path(args.output_path), matches= Path(args.output_path))
      
      # conf = extract_features.confs[args.feature]
      # extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path))
      
      ### Match feature of sequences
      # match_features.main(match_features.confs['superglue'], pairs = Path(args.pairs), features= Path(args.features), features_ref= Path(args.features_ref), matches= Path(args.output_path))
      
      ### loftr matches
      # conf = match_dense.confs['loftr']
      # match_dense.main(conf, Path(args.pairs), image_dir= Path(args.image_path), export_dir=Path(args.output_path))