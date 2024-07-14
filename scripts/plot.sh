#!/bin/bash
cd ..
python main.py --plot --eval_output_path dataset/GV-Bench/release/outputs/exp_day/single_pr_loftr.npy \
                        --exp_name day-single \
                        --method LoFTR \
                        --plot_save dataset/GV-Bench/release/outputs/exp_day/single_pr_loftr.pdf