from .. import logger


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
            querys.append(query)
            references.append(reference.split('/')[-1])
            labels.append(label)
    return querys, references, labels
            


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