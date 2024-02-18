from pathlib import Path
import pandas as pd
from tqdm import tqdm
from gvbench.utils import parse_pairs
from gvbench.evaluation import RANSAC, get_keypoints
from gvbench import logger
from get_final_dataset import cal_two_frame_dis_yaw
from multiprocessing import Process, Queue


# def stats(queue, pair, query_dir, reference_dir, match_path, feature_path):
#       stats = []
#       for q, r, l in tqdm(pair, total = len(pair)):
#             q_t = q.split('/')[-1].split('.')[0]
#             r_t = r.split('/')[-1].split('.')[0]
#             view, dist = cal_two_frame_dis_yaw(query_dir, reference_dir, int(q_t), int(r_t))
#             pointMap, inliers = RANSAC(q, r, match_path, feature_path)
#             kpts0 = get_keypoints(feature_path, q)
#             kpts1 = get_keypoints(feature_path, r)
#             stats.append([q, r, l, len(pointMap), len(inliers), len(kpts0), len(kpts1), view, dist])
#       queue.put(stats)


def stats(queue, pair, match_path, feature_path):
      stats = []
      for q, r, l in tqdm(pair, total = len(pair)):
            pointMap, inliers = RANSAC(q, r, match_path, feature_path)
            kpts0 = get_keypoints(feature_path, q)
            kpts1 = get_keypoints(feature_path, r)
            stats.append([q, r, l, len(pointMap), len(inliers), len(kpts0), len(kpts1)])
      queue.put(stats)

# def stats_multiprocess(pairs_path, query_dataset, ref_dataset, match_path, feature_path, num_process=8):
#       pairs_loader = parse_pairs(pairs_path, True)
#       pairs = [(q,r,l) for q,r,l in pairs_loader]
#       pairs = [pairs[i::num_process] for i in range(num_process)]
#       q = Queue()
#       processes = []
#       results = []
      
#       for i in range(num_process):
#             p = Process(target=stats, args=(q, pairs[i], query_dataset, ref_dataset, match_path, feature_path))
#             processes.append(p)
#             p.start()
#       for p in processes:
#             results.extend(q.get())
#       for p in processes:
#             p.join()
      
#       return results


def stats_multiprocess(pairs_path, match_path, feature_path, num_process=8):
      pairs_loader = parse_pairs(pairs_path, True)
      pairs = [(q,r,l) for q,r,l in pairs_loader]
      pairs = [pairs[i::num_process] for i in range(num_process)]
      q = Queue()
      processes = []
      results = []
      
      for i in range(num_process):
            p = Process(target=stats, args=(q, pairs[i], match_path, feature_path))
            processes.append(p)
            p.start()
      for p in processes:
            results.extend(q.get())
      for p in processes:
            p.join()
      
      return results

if __name__ == "__main__":

      # feature_paths = ['dataset/robotcar/features/qAutumn_dbRain/sift.h5', 'dataset/robotcar/features/qAutumn_dbRain/superpoint.h5', 'dataset/robotcar/features/qAutumn_dbRain/disk.h5', 'dataset/robotcar/features/qAutumn_dbRain/disk.h5']
      # match_paths = ['dataset/robotcar/matches/qAutumn_dbRain/matches-sift-nn.h5', 'dataset/robotcar/matches/qAutumn_dbRain/matches-superpoint-nn.h5', 'dataset/robotcar/matches/qAutumn_dbRain/matches-disk-nn.h5', 'dataset/robotcar/matches/qAutumn_dbRain/matches-disk-lightglue.h5']
      
      logger.setLevel('WARNING')
      # match_path = Path('dataset/robotcar/matches/qAutumn_dbRain/matches-disk-nn.h5')
      # feature_path = Path('dataset/robotcar/features/qAutumn_dbRain/disk.h5')
      # image_pair_path = 'dataset/robotcar/gt/robotcar_qAutumn_dbRain_final.txt'
      image_pair_path = 'dataset/uacampus/cosplace_gt.txt'
      match_path = Path('dataset/uacampus/matches/cosplace/matches-disk-lightglue.h5')
      feature_path = Path('dataset/uacampus/features/disk.h5')
      
      stats = stats_multiprocess(image_pair_path, match_path, feature_path, 40)
      # pairs_loader = parse_pairs(image_pair_path, True)
      # pairs = [(q,r,l) for q,r,l in pairs_loader]
      # stats = []
      
      # for q, r, l in tqdm(pairs):
      #       q_t = q.split('/')[-1].split('.')[0]
      #       r_t = r.split('/')[-1].split('.')[0]
      #       view, dist = cal_two_frame_dis_yaw('dataset/robotcar/Autumn_val', 'dataset/robotcar/Rain_val', int(q_t), int(r_t))
      #       pointMap, inliers = RANSAC(q, r, match_path, feature_path)
      #       kpts0 = get_keypoints(feature_path, q)
      #       kpts1 = get_keypoints(feature_path, r)
      #       stats.append([q, r, l, len(pointMap), len(inliers), len(kpts0), len(kpts1), view, dist])
      
      df = pd.DataFrame(stats, columns=['query', 'reference', 'label', 'matches', 'inliers', 'kpts0', 'kpts1'])
      
      df.to_csv('dataset/uacampus/exps/disk-nn_stats.csv', index=True)