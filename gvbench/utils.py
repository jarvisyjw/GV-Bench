import cv2
from . import logger
from tqdm import tqdm


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


def parse_pairs(gt: str, allow_label = False):
      
    logger.info(f'Loading ground truth from {gt}')
    logger.debug(f'Allow label: {allow_label}')
    
    f = open(gt, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        else:
            line = line.split(', ')
            query, reference, label = line
            if allow_label:
                  yield query, reference, label
            else:
                  yield query, reference


def write_pairs(file: str, pairs: list):
    logger.info(f'Writing pairs to {file}')
    f = open(file, 'w')
    for pair in tqdm(pairs):
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


def write_to_pairs(gt: str, pairs: str):
    logger.info(f'Loading ground truth from {gt}')
    loader = parse_pairs(gt)
    gts = [(q,r) for q,r in loader]
    logger.info(f'Writing pairs to {pairs}')
    f = open(pairs, 'w')
    for gt in tqdm(gts):
        f.write(f'{gt[0]} {gt[1]}\n')