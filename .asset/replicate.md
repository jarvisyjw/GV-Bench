# Replicate Results in Paper

- **Note**  
we use fundamental matrix estimation for outlier reject and use default image resolution of 1024*640, which is different from the image-matching-model's processing. Therefore if you replicate the experiments using image-matching-models, you may get different results. In order to foster the research, we deprecated the previous image matchers derived from hloc.

- ### Replicate from the log
  We provide the [output results](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jyubt_connect_ust_hk/EkflAPp79spCviRK5EkSGVABrGncg-TfNV5I3ThXxzopLg?e=tu91Xn) with the format shown below. You can use these results directly.

  - `$seq_$feature_$match.log`
  - `$seq_$feature_$match.npy` # with following format

  ```python
  np.save(str(export_dir), {
    'prob': num_matches_norm,
    'qImages': qImages,
    'rImages': rImages,
    'gt': labels, 
    'inliers': inliers_list,
    'all_matches': pointMaps,
    'precision': precision, 
    'recall': recall, 
    'TH': TH,
    'average_precision': average_precision,
    'Max Recall': r_recall
    })
  ```

- ### Replicate from scratch
    To get standard feature detection and matching results, we proposed to use [hloc](https://github.com/cvg/Hierarchical-Localization).

  - Download the dataset sequences from [google drive](https://drive.google.com/file/d/1145hQb812E0HaPGekdpD04bEbjuej4Lx/view?usp=drive_link) and put it under the `dataset/` folder.

  - Extract and match feature using hloc.
    - Extract features: SIFT, SuperPoint, and DISK
      
      ```bash
      python third_party/Hierarchical-Localization/gvbench_utils.py config/${seq}.yaml --extraction 
      ```
    - Match features: SIFT-NN, SIFT-LightGlue (Not yet implemented), SuperPoint-NN, DISK-NN, SuperPoint-SuperGlue, SuperPoint-LightGlue, DISK-LightGlue, LoFTR
      ```bash
      # all methods except LoFTR
      python third_party/Hierarchical-Localization/gvbench_utils.py config/${seq}.yaml --matching

      # LoFTR is different from above methods thus
      python third_party/Hierarchical-Localization/gvbench_utils.py config/${seq}.yaml --matching_loftr
      ```
  - Image pairs files
    - We prepare pairs (GT) file for matching under `dataset/gt` foler.
    - Make sure to use the fork hloc for feature extraction and matching `https://github.com/jarvisyjw/Hierarchical-Localization.git -b gvbench`
  - Example of ${seq}.yaml
    
    ```yaml
    night.yaml
        extraction:
        image_path: dataset/images
        features: 
        output_path: dataset/features
        matching:
        pairs: dataset/gt/night.txt
        feature: 
        feature_path: dataset/features
        output_path: dataset/matches
        matcher: 
        matching_loftr:
        pairs: dataset/gt/night.txt
        image_path: dataset/images
        matches: dataset/matches
        features: dataset/features
    ```