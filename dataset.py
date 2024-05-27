from torch.utils.data import Dataset, dataloader
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import itertools

from utils import parse_pairs, read_image, plot_sequence, logger


class SeqPairsDataset(Dataset):
      def __init__(self, qImage_path: Path, rImage_path: Path, 
                   qSeq_file_path: str, rSeq_file_path: str, pairs_file_path: str, seqL = 5):
            self.seqL = seqL
            
            self.qImages_path = qImage_path
            self.rImages_path = rImage_path
            self.qImages = [image for image in qImage_path.iterdir()]
            self.rImages = [image for image in rImage_path.iterdir()]
            
            self.qSeq_file = pd.read_csv(qSeq_file_path)
            self.rSeq_file = pd.read_csv(rSeq_file_path)
            
            self.pairs = [(path2timestamp(q), path2timestamp(r), int(l)) for q, r, l in parse_pairs(pairs_file_path, allow_label=True)]
      
      
      def __len__(self):
            return len(self.pairs)
      
      
      def __getitem__(self, index):
            qTimestamp, rTimestamp, label = self.pairs[index]
            qRow = self.qSeq_file[self.qSeq_file['timestamp'] == qTimestamp]
            rRow = self.rSeq_file[self.rSeq_file['timestamp'] == rTimestamp]
            
            qSeq = [qRow[f'{i}'].values[0] for i in range(self.seqL)]
            rSeq = [rRow[f'{i}'].values[0] for i in range(self.seqL)]
            
            qImages = [f'{image}.jpg' for image in qSeq]
            rImages = [f'{image}.jpg' for image in rSeq]
            
            pairs = list(itertools.product(qImages, rImages))
            
            return f'{qTimestamp}.jpg', f'{rTimestamp}.jpg', pairs, label
            
            
class SeqDataset(Dataset):
      def __init__(self, qImage_path: Path, rImage_path: Path, 
                   qSeq_file_path: str, rSeq_file_path: str, pairs_file_path: str, seqL = 5):
            self.seqL = seqL
            
            self.qImages_path = qImage_path
            self.rImages_path = rImage_path
            self.qImages = [image for image in qImage_path.iterdir()]
            self.rImages = [image for image in rImage_path.iterdir()]
            
            self.qSeq_file = pd.read_csv(qSeq_file_path)
            self.rSeq_file = pd.read_csv(rSeq_file_path)
            
            self.pairs = [(path2timestamp(q), path2timestamp(r), int(l)) for q, r, l in parse_pairs(pairs_file_path, allow_label=True)]
      
      def __len__(self):
            return len(self.pairs)
      
      
      def __getitem__(self, index):
            qTimestamp, rTimestamp, label = self.pairs[index]
            
            qRow = self.qSeq_file[self.qSeq_file['timestamp'] == qTimestamp]
            rRow = self.rSeq_file[self.rSeq_file['timestamp'] == rTimestamp]
            
            if qRow.empty or rRow.empty:
                  
                  return None
            else:
                  logger.info(f'qRow: {qRow}, rRow: {rRow}')
                  qSeq = [qRow[f'{i}'].values[0] for i in range(self.seqL)]
                  rSeq = [rRow[f'{i}'].values[0] for i in range(self.seqL)]
            
                  qImages = [read_image(self.qImages_path / f'{image}.jpg') for image in qSeq]
                  rImages = [read_image(self.rImages_path / f'{image}.jpg') for image in rSeq]
                  
                  return qImages, rImages, label


def path2timestamp(path: str):
      if not isinstance(path, Path):
            path = Path(path)
      return int(path.stem)


class EvaluationDataset(Dataset):
      
      def __init__(self, pairs_file: Path, image = False, qImage_path = None, rImage_path = None):
                        
            self.qImage_path = qImage_path
            self.rImage_path = rImage_path
            self.image = image
            
            self.pairs = [(q, r, int(l)) for q, r, l in parse_pairs(pairs_file, allow_label=True)]
            
      
      def __len__(self):
            return len(self.pairs)
      
      
      def __getitem__(self, index):            
            qName, rName, label = self.pairs[index]
            
            if self.image:
                  return read_image(self.qImage_path / Path(qName).name ), read_image(self.rImage_path / Path(rName).name ), label
            else:
                  return qName, rName, label


# class SingleImageDataset(Dataset):
#       '''
#       TODO: Implement the GVDataset class
#             For evaluation usage mainly,
#             Functions:
#             1. retrieve the image pairs for RANSAC candidate selection
#       '''
#       def __init__(self, Timestamp_list: str, poses_list: Path, pairs_file: Path):
                        
#             self.qImage_path = qImage_path
#             self.rImage_path = rImage_path
#             # self.pairs = [(path2timestamp(q),path2timestamp(r),int(l)) for q, r, l in parse_pairs(pairs_file, allow_label=True)]
#             self.pairs = [(q,r,int(l)) for q, r, l in parse_pairs(pairs_file, allow_label=True)]
            
#       def __len__(self):
#             return len(self.pairs)
      
#       def __getitem__(self, index):
#             qTimestamp, rTimestamp, label = self.pairs[index]
#             # qRow = self.seq_file[self.seq_file['timestamp'] == qTimestamp]
#             # rRow = self.seq_file[self.seq_file['timestamp'] == rTimestamp]
            
#             # qSeq = [qRow[f'{i}'].values[0] for i in range(self.seqL)]
#             # rSeq = [rRow[f'{i}'].values[0] for i in range(self.seqL)]
            
#             # qImages = [read_image(self.image_path / f'{image}.jpg') for image in qSeq]
#             # rImages = [read_image(self.image_path / f'{image}.jpg') for image in rSeq]
            
#             return qTimestamp.strip('.jpg').split('/')[-1], rTimestamp.strip('.jpg').split('/')[-1], label
#             # return f'{qTimestamp}.jpg', f'{rTimestamp}.jpg', label
      
#       # def _path2timestamp(self, path: str):
#       #       if not isinstance(path, Path):
#       #             path = Path(path)
#       #       return int(path.name.strip('.jpg'))
      

def test():
      qImage_path = Path('dataset/GV-Bench/release/images/day0_seq')
      rImage_path = Path('dataset/GV-Bench/release/images/night0_seq')
      qSeq_file_path = 'dataset/GV-Bench/release/sequences/day0/sequences_5.csv'
      rSeq_file_path = 'dataset/GV-Bench/release/sequences/night0/sequences_5.csv'
      pairs_file_path = 'dataset/GV-Bench/release/gt/night.txt'
      
      dataset = SeqDataset(qImage_path, rImage_path, qSeq_file_path, rSeq_file_path, pairs_file_path)
      qImages, rImages, Label = dataset[0]
      plot_sequence([qImages, rImages])
      plt.show()


def parser():
      from argparse import ArgumentParser
      parser = ArgumentParser()
      parser.add_argument('--qImage_path', type=str, help='Path to the query image folder')
      parser.add_argument('--rImage_path', type=str, help='Path to the reference image folder')
      parser.add_argument('--qSeq_file_path', type=str, help='Path to the query sequence file')
      parser.add_argument('--rSeq_file_path', type=str, help='Path to the reference sequence file')
      parser.add_argument('--pairs_file_path', type=str, help='Path to the pairs file')
      parser.add_argument('--output_path', type=str, help='Path to the output file')
      return parser.parse_args()

def write_pairs(args):
      dataset = SeqPairsDataset(Path(args.qImage_path), Path(args.rImage_path), args.qSeq_file_path, args.rSeq_file_path, args.pairs_file_path)
      #Iterate over each pairs of dataset and write the qImage and rImage name to a txt file
      with open(args.output_path, 'w') as f:
            for idx, pairs in tqdm(enumerate(dataset), total= len(dataset)):
                  for qImage, rImage in pairs:
                        f.write(f'{qImage} {rImage}\n')
      f.close()
      logger.info(f'{len(dataset)*5*5} pairs written to {args.output_path}')
                  

# visualize the dataset for testing
if __name__ == "__main__":
      pass
      # write_pairs(parser())
      # qImage_path = Path('dataset/GV-Bench/release/images/day0_seq')
      # rImage_path = Path('dataset/GV-Bench/release/images/night0_seq')
      # qSeq_file_path = 'dataset/GV-Bench/release/sequences/day0/sequences_5.csv'
      # rSeq_file_path = 'dataset/GV-Bench/release/sequences/night0/sequences_5.csv'
      # pairs_file_path = 'dataset/GV-Bench/release/gt/night.txt'
      
      # dataset = SeqDataset(qImage_path, rImage_path, qSeq_file_path, rSeq_file_path, pairs_file_path)
      # qImages, rImages, Label = dataset[500]
      # plot_sequence([qImages, rImages])
      # plt.show()
      # plt.show()
      # plt.imshow(qImages[0])
      # plt.show()
      # plot_sequence(qImages)
      # plt.show()
      # plot_sequence(rImages)
      # plt.show()
      # print(dataset[0])
                  
                  
            