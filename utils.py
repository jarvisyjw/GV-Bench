from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
import bisect
import csv
import numpy as np
import numpy.matlib as ml
import warnings
import matplotlib.pyplot as plt
import cv2
import shutil
import h5py
from typing import Tuple


from transform import se3_to_components
from interpolate_poses import interpolate_ins_poses


### logger
formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger("GV-Bench")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False


### Crop images
def crop_image(image_dir: str):
    im = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    h, w , c = im.shape
    im = im[:h-160, :, :]
    return im


### Some of the functions are copied from hloc
def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))

def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')


def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
        # print('uncertaintylistlens', len(uncertainty))
    if return_uncertainty:
        return p, uncertainty
    return p


def get_matches(path: Path, name0: str, name1: str, out=None) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    if out is not None:
        matches_score = np.column_stack((matches, scores))
        with open(out, 'w') as f:
            f.write('\n'.join(' '.join(map(str, match)) for match in matches_score))
        f.close()
    return matches, scores


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True, figsize=4.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*figsize, figsize]
    fig, axs = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)
    

def plot_sequence(images, figsize=(15, 10), dpi=100, pad=.5, show = True, label = None):
    """Plot a set of image sequences where horizontal images are from the same sequence.
    Args:
        images: a list of list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        
    """
    n = len(images) # num of sequences
    l = len(images[0]) # num of images in each sequence
    
    fig, axs = plt.subplots(n, l, figsize=figsize, dpi=dpi)
    
    if n == 1:
        axs = [axs]
    for i, (image, ax) in enumerate(zip(images, axs)):
        for j, img in enumerate(image):
            ax[j].imshow(img)
            # ax[j].set_title(f'Image {j}')
        # ax.imshow(image)
            ax[j].set_axis_off()
    fig.tight_layout(pad=pad)
    
    if label is not None:
        fig.suptitle(f'This is a {label} pair.', fontsize=16)
    if show:
        plt.show()


def parse_pairs(file: Path, allow_label = False):

    logger.info(f'Loading ground truth from {file}')
    logger.info(f'Allow label: {allow_label}')
    if isinstance(file, Path):
          file = str(file)
    
    f = open(file, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(' ')
            if allow_label:
                  query, reference, label = line
                  yield query, reference, label
            else:
                  query, reference = line
                  yield query, reference
    f.close()


def parse_timestamps(file: str):
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line.startswith('#'):
                continue
            else:
                yield line
    

def interpolate_poses(image_path: Path, gps_file: str, output: str):
    
    image_t = sorted([int(image.name.strip('.jpg')) for image in image_path.iterdir()])
    
    ins_pd = pd.read_csv(gps_file)
    ts = ins_pd['timestamp'].tolist()
    t_ref = ts[0]

    dataframe = []
    _, poses = interpolate_ins_poses(gps_file, image_t, t_ref)

    for t, pose in tqdm(zip(image_t, poses), total=len(image_t)):
        xyzrpy = se3_to_components(pose)
        # xyzrpy to list
        xyzrpy = xyzrpy.tolist()
        # print(xyzrpy, isinstance(xyzrpy, list))
        # np.set_printoptions(precision=12, suppress=True)
        out = [t, *xyzrpy]
        dataframe.append(out)

    df = pd.DataFrame(dataframe, columns=['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    df.to_csv(output, index=False)
    
    return df


def get_poses(poses_file: str):
    poses = [np.zeros(6)]
    timestamps = [0]
    
    with open(poses_file) as poses_file:
        poses_reader = csv.reader(poses_file)
        headers = next(poses_file)

        for row in poses_reader:
            timestamp = int(row[0])
            utm = row[1:4]
            rpy = row[-3:]
            xyzrpy = [float(v) for v in utm] + [float(v) for v in rpy]
            timestamps.append(timestamp)
            poses.append(xyzrpy)
    
    return timestamps[1:], poses[1:]


def dist_matrix(pose_file: str):
    pass



def generate_sequence(poses_file: str, origin: list, length: int, output_file: str):
    # TODO: out of boundary check
    # manually process now
    
    timestamps, poses = get_poses(poses_file)
    centers = [bisect.bisect(timestamps, t) for t in origin]
    
    logger.debug(f'Max idx: {max(centers)}')

    if max(centers) >= len(timestamps):
        centers = [min(i, len(timestamps) - 1) for i in centers]
    
    if length%2 == 0:
        raise ValueError('Length must be odd')
    
    seq_out = []
    
    for i in tqdm(centers):
        seq = find_sequence(timestamps, poses, i, length)
        seq_out.append(seq)
    
    frames = []
    errors = []
    for t, seq in zip(origin, seq_out):
        if seq is not None:
            frames.append([t] + seq)
        else:
            errors.append(t)
    
    columns = ['timestamp'] + [f'{i}' for i in range(length)]
    pd.DataFrame(frames, columns=columns).to_csv(output_file, index=False)
    
    return seq_out, errors


def find_sequence(timestamps: list, poses: list, idx: int, length: int):
    '''
    Find sequence of timestamps given the current timestamp
    '''
    
    seq = []
    forward_seq = []
    backward_seq = []

    # forward step
    steps = length//2
    
    logger.debug(f'Length: {length}, Steps: {steps}')
    logger.debug(f'Current index: {idx}')
    logger.debug(f'Length of all sequence: {len(timestamps)}')
    logger.debug(f'Current timestamp: {timestamps[idx]}')
    
    if check_dist(poses[idx], poses[len(timestamps)-1], distance_threshold=steps * 5) and check_dist(poses[idx], poses[0], distance_threshold=steps * 5):
        # search both forward and backward
        forward_seq = search(poses, timestamps, steps, idx, 1)
        backward_seq = search(poses, timestamps, steps, idx, -1)
        seq = backward_seq + [timestamps[idx-1]] + forward_seq
        # seq = backward_seq[::-1] + [timestamps[idx-1]] + forward_seq
    
    elif check_dist(poses[idx], poses[len(timestamps)-1], distance_threshold=(length-1) * 5):
        # search forward
        seq = search(poses, timestamps, length-1, idx, 1)
        seq = seq.insert(0, timestamps[idx-1])
        
    else:
        # search backwoard
        seq = search(poses, timestamps, length-1, idx, -1)
        seq = seq.append(timestamps[idx-1])
    
    return seq


def search(poses: list, timestamps: list, steps: int, center: int, direction = 1):
    # direction 1 for forward search
    # direction -1 for backward search
    seq = []
    pt = center + direction
    
    while steps > 0:
        if check_dist(poses[pt], poses[center]):
            if direction == -1:
                seq.insert(0, timestamps[pt])
            else:
                seq.append(timestamps[pt])
            center = pt
            pt = pt + direction
            steps -= 1
        else:
            pt = pt + direction
            
    return seq


def check_dist(xyzrpy, xyzrpy_ref, distance_threshold=5, large = True):
    '''
    Check if the distance between the current timestamp and the next timestamp is greater than 0.5s
    '''
    
    xyzrpy = np.array([float(p) for p in xyzrpy])
    xyzrpy_ref = np.array([float(p) for p in xyzrpy_ref])
    dist = np.linalg.norm(xyzrpy[:2] - xyzrpy_ref[:2])
    
    if large: 
        return False if dist < distance_threshold else True
    else: 
        return True if dist < distance_threshold else False
    

def pre_dataset(image_path: Path, sequence_file: str, dump_dir: Path):
    '''
    Prepare the dataset for the sequence
    '''
    df = pd.read_csv(sequence_file)
    for i in tqdm(range(len(df.columns)-1)):
        images = df[str(i)].tolist()
        for image in images:
            shutil.copyfile(Path(image_path, f'{image}.jpg'), Path(dump_dir, f'{image}.jpg'))
    
    logger.info(f'Dataset prepared at {dump_dir}')


def rm_keypoints(keypoints: Path, name: str):
    
    with h5py.File(str(keypoints), 'a') as hfile:
        if name in hfile:
            del hfile[name]
        else:
            logger.warning(f'Key {name} not found in {keypoints}')


def rm_matches(matches: Path, name0: str, name1: str):
    with h5py.File(str(matches), 'a') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        if pair in hfile:
            del hfile[pair]
        else:
            logger.warning(f'Pair {pair} not found in {matches}')
    
# def gen_match_pairs(pairs_file: str, output_file: str):
#     '''
#     Generate matching pairs for the sequence
#     '''
#     with open(pairs_file, 'r') as f:
#         pairs = f.readlines()
    
#     with open(output_file, 'w') as f:
#         for pair in pairs:
#             pair = pair.strip('\n').split(' ')
#             f.write(f'{pair[0]}.jpg {pair[1]}.jpg\n')
#     f.close()
#     logger.info(f'Matching pairs generated at {output_file}')