import logging
# import sys
# sys.path.append('../third_party/Hierarchical-Localization/')

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