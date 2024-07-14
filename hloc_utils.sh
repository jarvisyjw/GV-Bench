#!/bin/bash

First extract features
echo "Extracting features..."
python third_party/Hierarchical-Localization/gvbench_utils.py --extraction --image_path dataset/images --output_path dataset/features

# # Then match the features
# # day sequence
# echo "Matching features for day sequence..."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching --pairs dataset/gt/day.txt --features dataset/features --output_path dataset/matches

# # night sequence
# echo "Matching features for night sequence..."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching --pairs dataset/gt/night.txt --features dataset/features --output_path dataset/matches

# # season sequence
# echo "Matching features for season sequence..."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching --pairs dataset/gt/season.txt --features dataset/features --output_path dataset/matches

# # weather sequence
# echo "Matching features for weather sequence..."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching --pairs dataset/gt/weather.txt --features dataset/features --output_path dataset/matches

# night-hard sequence
# echo "Matching features for night-hard sequence..."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching --pairs dataset/gt/night-hard.txt --features dataset/features --output_path dataset/matches

# season-hard sequence
echo "Matching features for season-hard sequence..."
python third_party/Hierarchical-Localization/gvbench_utils.py --matching --pairs dataset/gt/season-hard.txt --features dataset/features --output_path dataset/matches

# Match dense using LoFTR
# echo "Matching featurs by LoFTR..."

# echo "Matching features for day sequence.."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching_loftr --pairs dataset/gt/day.txt --features dataset/features --output_path dataset/matches

# echo "Matching features for night sequence.."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching_loftr --pairs dataset/gt/night.txt --features dataset/features --output_path dataset/matches

# echo "Matching features for season sequence.."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching_loftr --pairs dataset/gt/season.txt --features dataset/features --output_path dataset/matches --image_path dataset/images

# echo "Matching features for weather sequence.."
# python third_party/Hierarchical-Localization/gvbench_utils.py --matching_loftr --pairs dataset/gt/weather.txt --features dataset/features --output_path dataset/matches --image_path dataset/images

echo "Matching features for night-hard sequence.."
python third_party/Hierarchical-Localization/gvbench_utils.py --matching_loftr --pairs dataset/gt/night-hard.txt --features dataset/features --output_path dataset/matches --image_path dataset/images

echo "Matching features for night-hard sequence.."
python third_party/Hierarchical-Localization/gvbench_utils.py --matching_loftr --pairs dataset/gt/season-hard.txt --features dataset/features --output_path dataset/matches --image_path dataset/images