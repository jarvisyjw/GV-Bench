from utils import interpolate_poses, generate_sequence, plot_images, read_image, search, get_poses, plot_sequence, pre_dataset, logger
from dataset import SeqPairsDataset, GVDataset, SeqDataset, EvaluationDataset
from eval import seqmatch, calpr, plot_pr_curve, Eval, max_recall, Eval_MP

from pathlib import Path
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import logging

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
      file_handler = logging.FileHandler(f'{args.output_path}/{Path(args.matches_path).stem}.log')
      file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      file_handler.setFormatter(file_formatter)
      file_handler.setLevel(logging.INFO)  # Set the desired log level for the file
      # Add the file handler to the existing logger
      logger.addHandler(file_handler)
      logger.setLevel('INFO')
      
      logger.info('Start Evaluation in Single Image Mode...')
      # All four sequences
      dataset = EvaluationDataset(pairs_file = args.pairs_file_path)
      # use multiple-process for acceleration
      Eval(dataset, Path(args.matches_path), 
              Path(args.features), 
              export_dir= Path(args.output_path, f'{Path(args.matches_path).stem}.npy'))

      # Eval_MP(10, dataset, Path(args.matches_path), 
      #         Path(args.features), 
      #         export_dir= Path(args.output_path, f'{Path(args.matches_path).stem}.npy'))



def eval(args):
      # dataset
      logger.info('Loading Dataset...')
      # dataset = GVDataset(Path(args.qImage_path), Path(args.rImage_path), args.pairs_file_path)
      dataset = SeqPairsDataset(Path(args.qImage_path), args.rImage_path, args.qSeq_file_path, args.rSeq_file_path, args.pairs_file_path, seqL = 5)
      matches_path = Path(args.matches_path)
      features = Path(args.features)
      features_ref = Path(args.features_ref)
      ransac_output = Path(args.ransac_output)
      # seq_match
      # logger.info('Single Matching')
      # singlematch(dataset, matches_path, features, features_ref, ransac_output)
      # logger.info('Sequence Matching...')
      seqmatch(dataset, matches_path, features, features_ref, ransac_output)
      logger.info('Sequence Matching Done...')
      logger.info('Calculating Precision and Recall...')
      precision, recall, average_precision = calpr(ransac_output, args.output_path)
      logger.info('Precision and Recall Calculation Done...')
      _, r_recall = max_recall(precision, recall)
      logger.info(f'\n' +
            f'Evaluation results: \n' +
            'Average Precision: {:.5f} \n'.format(average_precision) + 
            'Maximum Recall @ 100% Precision: {:.5f} \n'.format(r_recall))
      return None
      

def parser():
      parser = ArgumentParser()
      parser.add_argument('--poses_file', type=str)
      parser.add_argument('--image_path', type=Path)
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
      args = parser.parse_args()
      return args

def main():
      args = parser()
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
            logger.setLevel('INFO')
            logger.info('Generating Sequence...')
            dataset = GVDataset(Path(args.qImage_path), Path(args.rImage_path), args.pairs_file_path)
            # for idx, data in enumerate(dataset):
            #       qImages, rImages, label = data
            #       print(qImages)
            qImages_t = [int(data[0]) for idx, data in enumerate(dataset)]
            _, q_error = generate_sequence(args.poses_file, qImages_t, int(args.sequence_length), args.output_file)
            logger.info(f'Ambiguous Timestamps: {q_error}')
            
      
      # Interpolate poses
      if args.interpolate_poses:
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
                        qImages, rImages, label = dataset[image]
                        plot_sequence([qImages, rImages], label=label)
            
      
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