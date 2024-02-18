import pandas as pd
import math
import os.path as osp
from tqdm import tqdm
import multiprocessing

from .utils import parse_pairs_from_retrieval
from . import logger


def load_frame_utm(csv: str):
      '''
      input: csv file path
      output: utm_x, utm_y 
      '''
      a = pd.read_csv(csv)
      output_list = [i for i in a]
      if output_list[-2] == 'NaN' or output_list[-1] == 'NaN':
            return 0, 0
      else:
            return float(output_list[-2]), float(output_list[-1])


def cal_distance_utm(x1, y1, x2, y2):
      '''
      input: utm_x, utm_y
      output: distance [meters]
      '''
      if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            return -1
      else:
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def cal_gt_utm(root_dir: str, frame0: str, frame1: str):
      '''
      input: csv file path
      output: distance [meters]
      '''
      x1, y1 = load_frame_utm(osp.join(root_dir, frame0))
      x2, y2 = load_frame_utm(osp.join(root_dir, frame1))
      dist = cal_distance_utm(x1, y1, x2, y2)
      if dist == -1:
            return -1
      elif  dist < 25:
            return 1
      else:
            return 0

def main():
      pairs_path = 'dataset/tokyo247/pairs/cosplace.txt'
      output_path = 'dataset/tokyo247/pairs/cosplace_gt.txt'
      root_dir = 'dataset/tokyo247/images/'
      pairs_loader = parse_pairs_from_retrieval(pairs_path)
      pairs = [(q,r) for q,r in pairs_loader]
      fout = open(output_path, 'w')
      
      for pair in tqdm(pairs):
            q, r = pair
            print(q, r)
            q_csv = q.split('.')[0] + '.csv'
            r_csv = r.strip('.jpg') + '.csv'
            print(q_csv, r_csv)
            gt = cal_gt_utm(root_dir, q_csv, r_csv)
            print(q_csv, r_csv, gt)
            fout.write(f'{q}, {r}, {gt}\n')


if __name__ == '__main__':
      main()
      