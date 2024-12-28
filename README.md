# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

### [[Arxiv]](https://arxiv.org/abs/2407.11736) [[Project Page]](https://jarvisyjw.github.io/GV-Bench/) [[Intro in Chinese]](https://mp.weixin.qq.com/s/edUw7vLep0zmve0Uj3IzkQ)
<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<!-- <p align="center">
<img src="./assets/figs/radar-final-iros.png" width="600" alt="Description">
</p> -->

<!-- (Under construction, full codes and results comming soon!) Feel free to dorp me an email or leave an issue!

This repo contains the implementation of GV-Bench, aiming at providing a fair and accessible benchmark for geometric verification. We employ three datasets Oxford Robotcar, Nordland, and UAcampus, containing appearance changes over long period. -->
<!-- ## Visualization of Image Matching -->
## Abstract
Visual loop closure detection is an important module in visual simultaneous localization and mapping (SLAM), which associates current camera observation with previously visited places. Loop closures correct drifts in trajectory estimation to build a globally consistent map. However, a false loop closure can be fatal, so verification is required as an additional step to ensure robustness by rejecting the false positive loops. Geometric verification has been a well-acknowledged solution that leverages spatial clues provided by local feature matching to find true positives. Existing feature matching methods focus on homography and pose estimation in long-term visual localization, lacking references for geometric verification. To fill the gap, this paper proposes a unified benchmark targeting geometric verification of loop closure detection under long-term conditional variations. Furthermore, we evaluate six representative local feature matching methods (handcrafted and learning-based) under the benchmark, with in-depth analysis for limitations and future directions.

<!-- ### Run-time Analysis
We measure the runtime of six methods listed in Table I on NVIDIA GeForce RTX 3090 GPU and Intel i7-13700K CPU over 10K runs. The results are shown in figure below as inference time over performance, i.e., max recall @100 precision. We can conclude that the runtime of six local feature matching methods is at a millisecond level on a modern GPU. The choice can be made based on the trade-off between time efficiency and performance.
<p align="center">
<img src="./assets/figs/inference_time_vs_MR-crop.png" width="300" alt="Description">
</p> -->

## News
- :star: Paper is release on [arxiv](https://arxiv.org/abs/2407.11736).
- :tada: The paper is accepted by IROS 2024!
- :rocket: Releasing the visualization of [image matching](./assets/appendix.pdf) results. ([google drive](https://drive.google.com/file/d/1145hQb812E0HaPGekdpD04bEbjuej4Lx/view?usp=drive_link))
- :rocket: Releasing the benchmark (easy)! Checkout the image pairs from `dataset/gt` and images from [google drive](https://drive.google.com/drive/folders/1E8m353fi3hv-gaytJuRPLhFeNLPWTak6?usp=sharing)

<!-- 
## Release Timeline
- [x] Appendix for visualization
  - [x] Visualization of image matches
  - [x] Visualization of inliers' distirbution (SP.+SG.)
    <p align="center">
    <img src="./assets/figs/spsg-4seqs.png" width="500" alt="Description">
    </p>
- [ ] Release benchmark sequences.
  - [x] Benchmark-easy (Day, Night, Weather, Season) 
    [x] Day
    - [x] Weather
    - [x] Night-easy
    - [x] Season-easy
  - [ ] Benchmark-hard (For sever viewpoint and conditional variations.)
- [x] Release Local feature extraction and matching implementation
- [x] Release evaluation tools
- [x] Release data analysis tools
- [ ] Expansion to other verification methods (TODO)
  - [x] Dopplergangers
  <!-- - [ ] Semantics
  - [ ] Keypoint topology
- [ ] Release sequence version of benchmark (TODO) -->



## Installation
We use part of the HLoc code for feature extraction and matching.  
```bash
git clone && cd GV-Bench
git submodule init
git submodule update
cd third_party/Hierarchival-Localization
git checkout gvbench # this is a customized fork version
python -m pip install -e .
```
## Replicate Results in Exps
We provide the [output results]() with the format shown below. You can use these results directly.
```bash
$seq_$feature_$match.log
$seq_$feature_$match.npy # with following format
```
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
### Replicate from scratch
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
  <!-- - We also provide the easy to run scripts
    ```bash
    cd scripts/
    bash evaluation.sh ${sequence_name}
    ``` -->
  - Image pairs files
    - We prepare pairs (GT) file for matching under `dataset/gt` foler.
    - Make sure to use the fork hloc for feature extraction and matching `https://github.com/jarvisyjw/Hierarchical-Localization.git -b gvbench`

- Evaluation
  - We provide out-of-box scripts
  
  ```bash
  cd GV-Bench/scripts
  bash ./evaluation <day> # run script with 
  #sequence name: day, night, weather, season, nordland, uacampus
  ```

- Visualization
  - Demos are presented in `plot_data.ipynb`

## TODO
- [ ] Enabling using customized local features for geometric verification (GV).

## Acknowledgement
- This work builds upon [hloc](https://github.com/cvg/Hierarchical-Localization), thanks for their amazing work.
- Contact: `jingwen.yu@connect.ust.hk`

