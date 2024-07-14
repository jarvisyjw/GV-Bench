#!/bin/bash
cd ..
# loop i from 1 to 10
for i in {0..5}
do
    python main.py --eval --qImage_path dataset/GV-Bench/release/images/day0 \
                      --rImage_path dataset/GV-Bench/release/images/day1 \
                      --pairs_file_path dataset/GV-Bench/release/gt/day.txt \
                      --features dataset/GV-Bench/release/outputs/superpoint-day0.h5 \
                      --features_ref dataset/GV-Bench/release/outputs/superpoint-day1.h5 \
                      --matches_path dataset/GV-Bench/release/outputs/sp_sg_day_seq_5.h5 \
                      --ransac_output dataset/GV-Bench/release/outputs/exp_day/single_ransac_$i.txt \
                      --output_path dataset/GV-Bench/release/outputs/exp_day/single_pr_$i
done

# python main.py --eval --qImage_path dataset/GV-Bench/release/images/day0 \
#                   --rImage_path dataset/GV-Bench/release/images/day1 \
#                   --pairs_file_path dataset/GV-Bench/release/gt/day.txt \
#                   --features dataset/GV-Bench/release/outputs/superpoint-day0.h5 \
#                   --features_ref dataset/GV-Bench/release/outputs/superpoint-day1.h5 \
#                   --matches_path dataset/GV-Bench/release/outputs/sp_sg_day_seq_5.h5 \
#                   --ransac_output dataset/GV-Bench/release/outputs/exp_day/single_ransac_h.txt \
#                   --output_path dataset/GV-Bench/release/outputs/exp_day/single_pr_h