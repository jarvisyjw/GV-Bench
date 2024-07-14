from hloc.utils.io import list_h5_names
from dataset import EvaluationDataset

print(list_h5_names('dataset/matches_seq/season_superpoint_max_superglue.h5'))
# # # print(list_h5_names('dataset/GV-Bench/release/outputs/superpoint.h5'))
# # dataset = EvaluationDataset(pairs_file='dataset/gt/night.txt')
# # print(dataset[10])
# # # pairs_list = [dataset[i::10] for i in range(10)]
# # # print(pairs_list[0])

# clean up images
# import glob
# import os
# image_path = '/mnt/DATA_JW/dataset/robotcar_tmp/2015-02-03-08-45-10/stereo/centre'
# images_delete = glob.glob(image_path + '/*.png')
# for image in images_delete:
#       os.remove(image)

# remove names from h5 file
# import h5py
# from pathlib import Path

# kpts_path = 'dataset/features/superpoint_max.h5'
# image_path = Path('dataset/images/season0')
# names = [image for image in image_path.iterate]