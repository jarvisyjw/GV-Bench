import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from utils import get_matches, get_keypoints, read_image, plot_images
from eval import RANSAC

def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.0):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        for i in range(len(kpts0)):
            fig.add_artist(
                matplotlib.patches.ConnectionPatch(
                    xyA=(kpts0[i, 0], kpts0[i, 1]),
                    coordsA=ax0.transData,
                    xyB=(kpts1[i, 0], kpts1[i, 1]),
                    coordsB=ax1.transData,
                    zorder=1,
                    color=color[i],
                    linewidth=lw,
                    alpha=a,
                )
            )

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
):
    ax = plt.gcf().axes[idx]
    t = ax.text(
        *pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def colorsInliers(inliers):
    colors = np.zeros((len(inliers), 3))
    colors[inliers, 0] = 1
    colors[~inliers, 1] = 1
    return colors


def viz_inliers_distribution(exp_log_path: Path, bins = None, save = True):
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
      

def find_image_pairs(exp_log_path: Path, prob: list, label: int):
      # find image pairs within the probability range and label
      
      exp_log = np.load(exp_log_path, allow_pickle=True).tolist()
      probs = np.array(exp_log['prob']).flatten()
      labels = np.array(exp_log['gt']).flatten()
      qImages = np.array(exp_log['qImages'])
      tImages = np.array(exp_log['rImages'])
      
      idx_labels = np.where(labels == label)
      idx_prob = np.intersect1d(np.where(probs >= prob[0]), np.where(probs <= prob[1]))
      idx = np.intersect1d(idx_labels, idx_prob)
      # filter out zero probs
    #   idx = idx[probs[idx] > 0]
      
      return idx, qImages[idx], tImages[idx], probs[idx]


def plot_matches_pair(match_path: Path, 
                 feature_path: Path, 
                 database_image: Path, 
                 image0: str, image1: str,
                 label: int, prob: float, distance = None):
      
      points, inliers = RANSAC(image0, image1, match_path, feature_path)
      
      image_0 = read_image(database_image / image0)
      image_1 = read_image(database_image / image1)
      plot_images([image_0, image_1], dpi=100)
      
      if distance is not None:
          add_text(1, f'Tanslation:{distance[0]:.2f}, Rotation: {distance[1]:.2f}', pos=(0.1,0.05))
      
      if len(points) == 0:
            add_text(0, f'{len(inliers)} inliers, {len(points)} matches', pos=(0.1,0.05))
            add_text(0, f'Label: {label}, Prob: {prob}', pos=(0.1,0.01))
            plt.show()
      else:
        if len(points) == len(inliers):
                    color = [0,1,0]
                    color = [color]*len(points)
        else:
                color = [[0,1,0] if i in inliers else [1,0,0] for i in points]
      
        plot_matches(points[:,:2], points[:,2:], ps=40, a = 0.1, color=color)
        add_text(0, image0)
        add_text(1, image1)
        add_text(0, f'{len(inliers)} inliers, {len(points)} matches', pos=(0.1,0.05))
        add_text(0, f'Label: {label}, Prob: {prob:.5f}', pos=(0.1,0.1))
        # plt.show()
        

def draw_matches(match_path: Path, feature_path: Path, database_image: Path, image0: str, image1: str):
        image_0 = read_image(database_image / image0)
        image_1 = read_image(database_image / image1)
        kpt0 = get_keypoints(feature_path, image0)
        kpt1 = get_keypoints(feature_path, image1)
        plot_images([image_0, image_1], dpi=100)
        matches, scores = get_matches(match_path, image0, image1)
        matches = matches[scores > 0.5]
        matches0 = matches[:,0]
        matches1 = matches[:,1]
        kpt0 = kpt0[matches0]
        kpt1 = kpt1[matches1]
        plot_matches(kpt0, kpt1, ps=40, a = 0.1)
        add_text(0, image0)
        add_text(1, image1)
        add_text(0, f'{match_path.stem}', pos=(0.01,0.05))
    
      
if __name__ == '__main__':
      exp_log_path = Path('dataset/exp_seq/night_superpoint_max_superglue.npy')
      viz_inliers_distribution(exp_log_path)