#!/bin/bash
cd ..

echo 'Evaluating single pair'
echo 'Evaluating the day sequence'

echo 'SIFT-NN'

python main.py --eval_single --pairs_file_path dataset/gt/day.txt \
                 --features dataset/features/sift.h5 \
                 --matches_path dataset/matches/day_sift_NN.h5 \
                 --output_path dataset/exps 

echo 'SuperPoint-NN'

python main.py --eval_single --pairs_file_path dataset/gt/day.txt \
                 --features dataset/features/superpoint_max.h5 \
                 --matches_path dataset/matches/day_superpoint_max_NN.h5 \
                 --output_path dataset/exps 

echo 'SuperPoint-SuperGlue'

python main.py --eval_single --pairs_file_path dataset/gt/day.txt \
                 --features dataset/features/superpoint_max.h5 \
                 --matches_path dataset/matches/day_superpoint_max_superglue.h5 \
                 --output_path dataset/exps 

echo 'DISK-NN'

python main.py --eval_single --pairs_file_path dataset/gt/day.txt \
                 --features dataset/features/disk.h5 \
                 --matches_path dataset/matches/day_disk_NN.h5 \
                 --output_path dataset/exps 

echo 'DISK-Lightglue'

python main.py --eval_single --pairs_file_path dataset/gt/day.txt \
                 --features dataset/features/disk.h5 \
                 --matches_path dataset/matches/day_disk_lightglue.h5 \
                 --output_path dataset/exps 

echo 'LoFTR'

python main.py --eval_single --pairs_file_path dataset/gt/day.txt \
                 --features dataset/features/day_loftr_kpts.h5 \
                 --matches_path dataset/matches/day_loftr.h5 \
                 --output_path dataset/exps