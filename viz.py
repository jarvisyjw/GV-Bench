import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

def viz_inliers_distribution(exp_log_path: Path, bins = None):
      exp_log = np.load(exp_log_path, allow_pickle=True).tolist()
      inliers = np.array(exp_log['prob']).flatten()
      labels = np.array(exp_log['gt']).flatten()
      # print(inliers.shape, labels.shape)
      # print(labels)
      # print(inliers)
      cols = np.stack([inliers, labels], axis=1)
      plt.rcParams['text.usetex'] = True
      # font times new roman
      # font bold
      plt.rcParams['font.weight'] = 'bold'
      plt.rcParams['font.family'] = 'Arial'
      # font size 10
      plt.rcParams['font.size'] = 12
      plt.figure(figsize=(4, 2.5))
      col0 = cols[cols[:,1]==0]
      col1 = cols[cols[:,1]==1]

      if bins is None:
            bins = [0,0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,1]
            
      plt.hist(col0[:,0], bins, density=True, cumulative=False, color='dodgerblue', histtype='bar', label='Non Loop Closure', alpha=0.7)
      plt.hist(col1[:,0], bins, density=True, cumulative=False, histtype='bar', color='orange', label='Loop Closure', alpha=0.75)
      plt.xlabel('Inliers Count')
      plt.ylabel('Density of Frequency')
      plt.legend()
      plt.tight_layout()
      plt.savefig(f'{exp_log_path.parent}/{exp_log_path.stem}.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
      

if __name__ == '__main__':
      exp_log_path = Path('dataset/exp_seq/night_superpoint_max_superglue.npy')
      viz_inliers_distribution(exp_log_path)