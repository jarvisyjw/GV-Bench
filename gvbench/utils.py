import cv2
from .. import logger


def crop_image(image_dir: str):
    im = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    h, w , c = im.shape
    im = im[:h-160, :, :]
    return im


def load_gt(gt: str):
    querys, references, labels = [], [], []
    logger.info(f'Loading ground truth from {gt}')
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            querys.append(query.split('/')[-1])
            references.append(reference.split('/')[-1])
            labels.append(label)
    return querys, references, labels


def parse_pairs(gt: str):
    logger.info(f'Loading ground truth from {gt}')
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            yield query, reference, label


def write_pairs(file: str, pairs: list):
    logger.info(f'Writing pairs to {file}')
    f = open(file, 'w')
    for pair in pairs:
        f.write(f'{pair[0]}, {pair[1]}, {pair[2]}\n')
    logger.info(f'Wrote pairs to {file}. DONE!')
            

def gt_loader(gt: str):
    logger.info(f'Loading ground truth from {gt}')
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            yield set(query, reference, label)