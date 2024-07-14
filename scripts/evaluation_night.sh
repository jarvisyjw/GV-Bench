#!/bin/bash
cd ..
python main.py --eval --qImage_path dataset/GV-Bench/release/images/day0_seq \
                      --rImage_path dataset/GV-Bench/release/images/night0_seq \
                      --qSeq_file_path dataset/GV-Bench/release/sequences/day0/sequences_5.csv \
                      --rSeq_file_path dataset/GV-Bench/release/sequences/night0/sequences_5.csv \
                      --pairs_file_path dataset/GV-Bench/release/gt/night.txt \
                      --matches_path dataset/GV-Bench/release/outputs/sp_sg_night_seq_5.h5 \
                      --features dataset/GV-Bench/release/outputs/superpoint-day0.h5 \
                      --features_ref dataset/GV-Bench/release/outputs/superpoint-night0.h5 \
                      --ransac_output dataset/GV-Bench/release/outputs/exp_night/seq_5_ransac.txt \
                      --output_path dataset/GV-Bench/release/outputs/exp_night/seq_5_pr