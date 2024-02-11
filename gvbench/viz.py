"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
# import cv2
from .evaluation import RANSACwithF
from .utils import RANSAC

from hloc.utils.io import read_image, get_keypoints, get_matches
from . import logger

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHILE = (255, 255, 255)


def name_to_pair(name: str):
    return name.strip('.jpg').replace('/', '-')


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 0.1, 1, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)
    return out


def cm_prune(x_):
    """Custom colormap to visualize pruning"""
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    max_i = max(x_)
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)
    return cm_BlRdGn(norm_x)


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: list of NumPy RGB (H, W, 3) or PyTorch RGB (3, H, W) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    # conversion to (H, W, 3) for torch.Tensor
    imgs = [
        img.permute(1, 2, 0).cpu().numpy()
        if (isinstance(img, torch.Tensor) and img.dim() == 3)
        else img
        for img in imgs
    ]

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
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
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

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


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
    

def plot_matches_from_pair(image0: str, image1: str, match_path: Path, feature_path: Path, database_image: Path, dpi=75, save_dir = None, label=None, ransac=True):
    logger.info(f'Plot matches of {image0} and {image1}')
    
    matches, scores = get_matches(match_path, image0, image1)
    # matches = matches[scores > 0.5]
    matches0 = matches[:,0]
    matches1 = matches[:,1]

    keypoint0 = get_keypoints(feature_path, image0)
    keypoint1 = get_keypoints(feature_path, image1)
    
    kp_0 = keypoint0[matches0]
    kp_1 = keypoint1[matches1]
    
    if ransac:
        # RANSAC
        points, inliers = RANSAC(kp_0, kp_1)
    else:
        points = np.array([kp_0, kp_1])
        inliers = points
        
        
    logger.debug(f'Keypoints of {image0}: {keypoint0.shape[0]}')
    logger.debug(f'Keypoints of {image1}: {keypoint1.shape[0]}')
    logger.debug(f'Number of matches: {kp_0.shape[0]}')
    
    
    if kp_0.shape[0] < 1 or kp_1.shape[0] < 1:
        logger.warning(f'No matches found for {image0} and {image1}')
        pass
    
    else:

        image_0 = read_image(database_image / image0)
        image_1 = read_image(database_image / image1)

        plot_images([image_0, image_1], dpi=dpi)
        
        if len(points) == len(inliers):
            color = [0,1,0]
        else:
            logger.debug(f'points: {len(points)}, inliers: {len(inliers)}')
            logger.debug(f'{points}, {inliers}')
            color = [[0,1,0] if i in inliers else [1,0,0] for i in points]
            logger.debug(f'Color: {color}')
        
        logger.debug(f'color: {len(color)}')
        plot_matches(points[:,:2], points[:,2:], ps=20, a = 0.1, color=color)
        add_text(0, image0)
        add_text(1, image1)
        add_text(0, f'{len(inliers)} inliers, {len(points)} matches', pos=(0.1,0.01))
        
        if label is not None:
            add_text(0, f'{label}', pos=(0.01,0.01))
            
        if save_dir is not None:
            logger.info(f'Save image at {str(save_dir)}')
            if isinstance(save_dir, str):
                save_dir = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            save_plot(save_dir / f'{name_to_pair(image0)}-{name_to_pair(image1)}.jpg')
        else:
            plt.show()


def visualize_FP(pairs_idx: np.ndarray, matches_path: Path, feature_path: Path, image_dir: str, save_dir: Path):
    for pair in tqdm(pairs_idx):
        query, reference = pair[0], pair[1]
        plot_matches_from_pair(query, reference, matches_path, feature_path, image_dir, 75, save_dir)


# def plot_demo(image0: str, image1: str, match_path: Path, feature_path: Path, database_image: Path, dpi=75):
#     logger.info(f'Plot matches of {image0} and {image1}')
    
#     matches, _ = get_matches(match_path, image0, image1)
#     matches0 = matches[:,0]
#     matches1 = matches[:,1]
    
#     keypoint0 = get_keypoints(feature_path, image0)
#     keypoint1 = get_keypoints(feature_path, image1)

#     kp_0 = keypoint0[matches0]
#     kp_1 = keypoint1[matches1]

#     image_0 = read_image(database_image / image0)
#     image_1 = read_image(database_image / image1)
    
#     plot_images([image_0, image_1], dpi=dpi)
    
#     # draw lines
#     for x1, y1, x2, y2 in zip(kp_0[:,0], kp_0[:,1], kp_1[:,0], kp_1[:,1]):
#         point1 = (int(x1), int(y1))
#         point2 = (int(x2 + image1.shape[1]), int(y2))
#         color = BLUE

#         cv2.line( plt.gcf(), point1, point2, color, 1)

#     # Draw circles on top of the lines
#     for x1, y1, x2, y2 in small_point_map:
#         point1 = (int(x1), int(y1))
#         point2 = (int(x2 + image1.shape[1]), int(y2))
#         cv2.circle(matchImage, point1, 2, BLUE, 2)
#         cv2.circle(matchImage, point2, 2, BLUE, 2)

