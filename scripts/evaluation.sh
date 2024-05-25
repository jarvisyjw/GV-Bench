#!/bin/bash
cd ..

seq=$1

echo 'Evaluating single pair on Sequence': $seq
echo 'Evaluating the day sequence'

echo 'SIFT-NN'

python main.py --eval_single --pairs_file_path dataset/gt/$seq.txt \
                 --features dataset/features/sift.h5 \
                 --matches_path dataset/matches/${seq}_sift_NN.h5 \
                 --output_path dataset/exps 

echo 'SuperPoint-NN'

python main.py --eval_single --pairs_file_path dataset/gt/$seq.txt \
                 --features dataset/features/superpoint_max.h5 \
                 --matches_path dataset/matches/${seq}_superpoint_max_NN.h5 \
                 --output_path dataset/exps 

echo 'SuperPoint-SuperGlue'

python main.py --eval_single --pairs_file_path dataset/gt/$seq.txt \
                 --features dataset/features/superpoint_max.h5 \
                 --matches_path dataset/matches/${seq}_superpoint_max_superglue.h5 \
                 --output_path dataset/exps 

echo 'DISK-NN'

python main.py --eval_single --pairs_file_path dataset/gt/$seq.txt \
                 --features dataset/features/disk.h5 \
                 --matches_path dataset/matches/${seq}_disk_NN.h5 \
                 --output_path dataset/exps 

echo 'DISK-Lightglue'

python main.py --eval_single --pairs_file_path dataset/gt/$seq.txt \
                 --features dataset/features/disk.h5 \
                 --matches_path dataset/matches/${seq}_disk_lightglue.h5 \
                 --output_path dataset/exps 

echo 'LoFTR'

python main.py --eval_single --pairs_file_path dataset/gt/$seq.txt \
                 --features dataset/features/${seq}_loftr_kpts.h5 \
                 --matches_path dataset/matches/${seq}_loftr.h5 \
                 --output_path dataset/exps