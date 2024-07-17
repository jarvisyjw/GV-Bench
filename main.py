from utils import *
from viz import *
from dataset import SeqPairsDataset, SeqDataset, EvaluationDataset
from eval import plot_pr_curve, Eval

from pathlib import Path
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import logging
from tqdm import tqdm

def append_df(df: pd.DataFrame, timestamps: list, seqs: list):
      assert len(timestamps) == len(seqs), "timestamps and seqs must have the same length"
      value = [[t] + seq for t, seq in zip(timestamps, seqs)]
      df_append = pd.DataFrame(value, columns=['timestamp'] + [f'{i}' for i in range(len(seqs[0]))])
      df_new = pd.concat([df, df_append], ignore_index=True)
      df_new = df_new.sort_values(by = 'timestamp')
      return df_new


def plot(args):
      # load matching results
      exp = np.load(args.eval_output_path, allow_pickle=True).item()
      precision = exp['precision']
      logger.debug(f'Precision: {precision}')
      recall = exp['recall']
      logger.debug(f'Recall: {recall}')
      average_precision = exp['average_precision']
      logger.info(f'Average Precision: {average_precision}')
      plot_pr_curve(recall, precision, average_precision, args.method, args.exp_name)
      if args.plot_save is not None:
            plt.savefig(args.plot_save, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
      plt.show()


def eval_single(args):
      '''
      Compute the PR curve for matching with single images.
      
      Args:
            args.pairs_file_path: str
            args.matches_path: str
            args.features: Path
            args.output_path: Path
      '''
      # logger setup
      # Create a file handler
      file_handler = logging.FileHandler(Path(args.output_path, f'{Path(args.matches_path).stem}.log'))
      file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      file_handler.setFormatter(file_formatter)
      file_handler.setLevel(logging.INFO)  # Set the desired log level for the file
      # Add the file handler to the existing logger
      logger.addHandler(file_handler)
      logger.setLevel('INFO')
      
      logger.info('Start Evaluation in Single Image Mode...')
      # All four sequences
      dataset = EvaluationDataset(pairs_file = args.pairs_file_path)
      # TODO: use multiple-process for acceleration
      Eval(dataset, Path(args.matches_path), 
              Path(args.features), 
              export_dir= Path(args.output_path, f'{Path(args.matches_path).stem}.npy'))


def crop_images(image_dir: str, export_dir: str, image_list = None):
      
      logger.info(f'Cropping images from {image_dir} and export to {export_dir}')
      image_dir = Path(image_dir)
      export_dir = Path(export_dir)
      
      if not export_dir.exists():
            export_dir.mkdir(parents=True, exist_ok=True)
      
      if image_list is not None:
            images = [image for image in image_dir.iterdir() if int(image.stem) in image_list]
            for image_path in tqdm(images, total= len(images)):
                  image = crop_image(str(image_path))
                  if not cv2.imwrite(str(export_dir / image_path.name), image):
                        raise Exception("Could not write image {}".format(export_dir / image_path.name))
                        
      else:
            images = [image for image in image_dir.iterdir()]
            
            for image_path in tqdm(images, total= len(images)):
                  image = crop_image(str(image_path))
                  if not cv2.imwrite(str(export_dir / image_path.name), image):
                        raise Exception("Could not write image {}".format(export_dir / image_path.name))
                  
      logger.info(f'Cropped images from {image_dir} to {export_dir}. DONE!')


def eval(args):
      '''Sequence Matching Evaluation

      '''
      '''
      Compute the PR curve for matching with single images.
      
      Args:
            args.output_path: Path
            args.matches_path: Path
            args.features: Path

            args.qImage_path: Path
            args.rImage_path: Path
            args.qSeq_file_path: Path
            args.rSeq_file_path: Path
            args.pairs_file_path: Path
      '''
      
      # logger setup
      # Create a file handler
      file_handler = logging.FileHandler(Path(args.output_path, f'{Path(args.matches_path).stem}.log'))
      file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      file_handler.setFormatter(file_formatter)
      file_handler.setLevel(logging.INFO)  # Set the desired log level for the file
      
      # Add the file handler to the existing logger
      logger.addHandler(file_handler)
      logger.setLevel('INFO')
      
      logger.info('Start Evaluation in Sequence Matching Mode...')
      # All four sequences
      dataset = SeqPairsDataset(args.qImage_path, args.rImage_path, args.qSeq_file_path,
                                args.rSeq_file_path, args.pairs_file_path, 5)
      
      # TODO: use multiple-process for acceleration
      Eval(dataset, Path(args.matches_path),
              Path(args.features),
              export_dir= Path(args.output_path, f'{Path(args.matches_path).stem}.npy'), 
              seq=True)
      

def parser():
      parser = ArgumentParser()
      parser.add_argument('--poses_file', type=str)
      parser.add_argument('--image_path', type=Path)
      parser.add_argument('--timestamp', type=str)
      parser.add_argument('--sequence_length', type=int)
      parser.add_argument('--gps_file', type=str)
      parser.add_argument('--output_file', type=str)
      parser.add_argument('--image_db', type=Path)
      parser.add_argument('--seq_file', type=Path)
      parser.add_argument('--dump_dir', type=Path)
      parser.add_argument('--qImage_path', type=Path)
      parser.add_argument('--rImage_path', type=Path)
      parser.add_argument('--qSeq_file_path', type=str)
      parser.add_argument('--rSeq_file_path', type=str)
      parser.add_argument('--pairs_file_path', type=str)
      parser.add_argument('--matches_path', type=str)
      parser.add_argument('--features', type=Path)
      parser.add_argument('--features_ref', type=Path)
      parser.add_argument('--ransac_output', type=Path)
      parser.add_argument('--output_path', type=Path)
      parser.add_argument('--eval_output_path', type=Path)
      parser.add_argument('--exp_log_path', type=Path)
      parser.add_argument('--plot_save', type=str)
      parser.add_argument('--exp_name', type=str)
      parser.add_argument('--method', type=str)
      parser.add_argument('--image_list', type=str)
      parser.add_argument('--eval', action='store_true')
      parser.add_argument('--eval_single', action='store_true')
      parser.add_argument('--plot', action='store_true')
      parser.add_argument('--gen_sequence', action='store_true')
      parser.add_argument('--interpolate_poses', action='store_true')
      parser.add_argument('--plot_sequence', action='store_true')
      parser.add_argument('--crop_images', action='store_true')
      parser.add_argument('--pre_seq_dataset', action='store_true')
      parser.add_argument('--gen_match_pairs', action='store_true')
      parser.add_argument('--plot_inliers_dist', action='store_true')
      args = parser.parse_args()
      return args

def main():
      
      args = parser()
      
      # prepare sequence dataset
      if args.pre_seq_dataset:
            '''Prepare Dataset
            Args:
                  args.image_path
                  args.output_path
                  args.seq_file
            '''
            pre_dataset(Path(args.image_path), args.seq_file, Path(args.output_path))

      
      if args.gen_match_pairs:
            '''Generate matching pairs
            Args:
                  args.qImage_path: Path
                  args.rImage_path: Path
                  args.qSeq_file_path: str
                  args.rSeq_file_path: str
                  args.pairs_file_path: str
                  args.output_path: str
            '''
            assert args.qImage_path is not None, 'Query Image path must be provided'
            assert args.rImage_path is not None, 'Reference Image path must be provided'
            assert args.qSeq_file_path is not None, 'Query Sequence file path must be provided'
            assert args.rSeq_file_path is not None, 'Reference Sequence file path must be provided'
            assert args.pairs_file_path is not None, 'Pairs file path must be provided'
            assert args.output_path is not None, 'Output path must be provided'

            dataset = SeqPairsDataset(Path(args.qImage_path), Path(args.rImage_path), 
                                      args.qSeq_file_path, args.rSeq_file_path, args.pairs_file_path, flag=True)
            
            with open(args.output_path, 'w') as f:
                  for idx, (pairs, label) in tqdm(enumerate(dataset), total= len(dataset)):
                        for pair in pairs:
                              qImage, rImage = pair
                              f.write(f'{qImage} {rImage}\n')
            f.close()

      
      # crop images
      if args.crop_images:
            '''Crop the car hood from the images
            Args:
                  args.image_path: Path
                  args.output_path: Path
            '''
            assert args.image_path is not None, 'Image path must be provided'
            assert args.output_path is not None, 'Output path must be provided'
            
            if args.image_list is not None:
                  image_file = open(args.image_list, 'r')
                  image_list = [int(image.strip()) for image in image_file.readlines() if not image.startswith('#')]
                  # print(image_list)
                  crop_images(args.image_path, args.output_path, image_list=image_list)
            
            else:
                  crop_images(args.image_path, args.output_path)
      
      
      # Evaluation mode
      if args.eval_single:
            logger.info('Evaluation Process Begins...')
            eval_single(args)
      
      # Plot mode
      if args.plot:
            logger.info('Plotting Images...')
            plot(args)
      
      # Evaluation in Single Image mode
      if args.eval:
            eval(args)
      
      # Generate sequence mode
      if args.gen_sequence:
            '''Generate Sequence
            Args:
                  args.poses_file: str
                  args.timestamp: str
                  args.sequence_length: int
                  args.output_file: str
            return: None
            '''
            assert args.poses_file is not None, 'Poses file must be provided'
            assert args.timestamp is not None, 'Timestamp must be provided'
            assert args.sequence_length is not None, 'Sequence length must be provided'
            assert args.output_file is not None, 'Output file must be provided'
            # logger setup
            logger.setLevel('INFO')
            logger.info('Generating Sequence...')
            loader = parse_timestamps(args.timestamp)
            Timestamp = [int(t) for t in loader] 
            seqs, errors = generate_sequence(args.poses_file, Timestamp, int(args.sequence_length), args.output_file)
            logger.info(f'Ambiguous Timestamps: {errors}')
            
      
      # Interpolate poses
      if args.interpolate_poses:
            '''Interpolate Poses via robotcar SDK
            Args:
                  args.image_path: Path
                  args.gps_file: str
                  args.output_file: str
            '''
            
            assert args.image_path is not None, 'Image path must be provided'
            assert args.gps_file is not None, 'GPS file must be provided'
            assert args.output_file is not None, 'Output file must be provided'
            interpolate_poses(args.image_path, args.gps_file, args.output_file)
      
      
      # Plot sequence
      if args.plot_sequence:
            
            dataset = SeqDataset(Path(args.qImage_path), Path(args.rImage_path), Path(args.qSeq_file_path), 
                                 Path(args.rSeq_file_path), args.pairs_file_path)
            
            if args.image_list is not None:
                  image_list = [int(image) for image in args.image_list.split(',')]
                  
                  
                  
            else:
                  image_list = [random.randint(0, len(dataset)-1) for i in range(5)]
                  
                  for image in image_list:
                        if dataset[image] is None:
                              continue
                        else:
                              qImages, rImages, label = dataset[image]
                              plot_sequence([qImages, rImages], label=label)
            
      
      # Plot inliers
      if args.plot_inliers_dist:
            
            viz_inliers_distribution(args.exp_log_path)
      
      
      '''
      Prepare the dataset for the sequence
      '''
      # pre_dataset(args.image_path, args.seq_file, args.dump_dir)
      
      
      '''
      Generate sequences
      '''
      # image_t = sorted([int(image.name.strip('.jpg')) for image in args.image_path.iterdir()])
      # _, error = generate_sequence(args.poses_file, image_t, args.sequence_length, args.output_file)
      # print(f'Ambiguous Timestamps: {error}')
      
      '''
      Mannually append sequences to the dataframe
      '''
      # df = pd.read_csv(args.seq_file)
      # forward_ts = [1418757503567203, 1418757503879662, 1418757504504577, 1418757504879527, 1418757508566532]
      # forward_seqs = [[1418757498567877, 1418757499630232, 1418757500880066, 1418757502192390, 1418757503567203],
      #                   [1418757498817843, 1418757499880200, 1418757501130031, 1418757502504848, 1418757503879662],
      #                   [1418757499255285, 1418757500380133, 1418757501692455, 1418757503067272, 1418757504504577],
      #                   [1418757499442760, 1418757500630099, 1418757501942423, 1418757503317238, 1418757504879527],
      #                   [1418757500630099, 1418757501942423, 1418757503317238, 1418757504942019, 1418757508566532]]
      # backward_ts = [1418134766291893, 1418134766541851, 1418134766791828, 1418134767041806, 1418134767291782, 1418134767604234, 1418134767916680, 1418134768291636, 1418134768666609, 1418134769104043, 1418134769603963, 1418134770166372, 1418134770791314, 1418134771603697, 1418134772603561, 1418134774103318, 1418134776665509]
      # backward_seqs = [[1418134762604895, 1418134763479748, 1418134764354604, 1418134765291966, 1418134766291893],
      #                   [1418134762729876, 1418134763604728, 1418134764479585, 1418134765479453, 1418134766541851],
      #                   [1418134762979833, 1418134763854688, 1418134764729542, 1418134765729431, 1418134766791828],
      #                   [1418134763104811, 1418134763979668, 1418134764917011, 1418134765916920, 1418134767041806],
      #                   [1418134763292278, 1418134764167137, 1418134765104481, 1418134766166904, 1418134767291782],
      #                   [1418134763542238, 1418134764417094, 1418134765354458, 1418134766416871, 1418134767604234],
      #                   [1418134763604728, 1418134764479585, 1418134765479453, 1418134766604340, 1418134767916680],
      #                   [1418134763854688, 1418134764729542, 1418134765729431, 1418134766854318, 1418134768291636],
      #                   [1418134763979668, 1418134764917011, 1418134765916920, 1418134767104306, 1418134768666609],
      #                   [1418134764167137, 1418134765104481, 1418134766166904, 1418134767354275, 1418134769104043],
      #                   [1418134764354604, 1418134765291966, 1418134766354382, 1418134767604234, 1418134769603963],
      #                   [1418134764479585, 1418134765479453, 1418134766604340, 1418134767916680, 1418134770166372],
      #                   [1418134764729542, 1418134765729431, 1418134766854318, 1418134768291636, 1418134770791314],
      #                   [1418134764854520, 1418134765854428, 1418134767041806, 1418134768604118, 1418134771603697],
      #                   [1418134765041991, 1418134766104406, 1418134767291782, 1418134769104043, 1418134772603561],
      #                   [1418134765229467, 1418134766291893, 1418134767541743, 1418134769541474, 1418134774103318],
      #                   [1418134765416955, 1418134766541851, 1418134767854191, 1418134770041392, 1418134776665509]]

      # timestamps = forward_ts
      # seqs = forward_seqs
      
      # df_new = append_df(df, timestamps, seqs)
      # df_new.to_csv(args.output_file, index=False)


      '''
      Mannually search for sequences
      '''
      # interpolate_poses(args.image_path, args.gps_file, args.output_file)
      
      # timestamps , poses = get_poses(args.poses_file)
      
      # forward_search = forward_search = [1416316985484274, 1416316986234142, 1416317013667830, 1416317014042806, 1416317014355275, 1416317014667735, 1416317014980179, 1416317015230153, 1416317015480119]
      # backward_search = backward_search = [1416319227487911, 1416319227925377, 1416319228362851, 1416319228862826, 1416319229487770, 1416319230237662, 1416319230925047, 1416319231674920, 1416319232549828, 1416319233549720, 1416319234924496, 1416319237299109]
      
      '''
      Backward Search
      '''
      # for t in backward_search:
      #       center = bisect.bisect(timestamps, t)
      #       seq = search(poses, timestamps, 4, center, -1)
      #       seq.append(timestamps[center-1])
      #       # seq.insert(0, timestamps[center-1])
      #       print(seq)
      #       plot_sequence([[read_image(image_path / f'{t}.jpg') for t in seq]])
      #       plt.show()
      
      '''
      Forward Search
      '''
      # for t in forward_search:
      #       center = bisect.bisect(timestamps, t)
      #       seq = search(poses, timestamps, 4, center, 1)
      #       # seq.append(timestamps[center-1])
      #       seq.insert(0, timestamps[center-1])
      #       print(seq)
      #       plot_sequence([[read_image(image_path / f'{t}.jpg') for t in seq]])
      #       plt.show()
      

if __name__ == '__main__':
    main()