# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<p align="center">
<img src="./assets/figs/radar-images.png" width="600" alt="Description">
</p>

<!-- (Under construction, full codes and results comming soon!) Feel free to dorp me an email or leave an issue!

This repo contains the implementation of GV-Bench, aiming at providing a fair and accessible benchmark for geometric verification. We employ three datasets Oxford Robotcar, Nordland, and UAcampus, containing appearance changes over long period. -->
<!-- ## Visualization of Image Matching -->
## Abstract
Visual loop closure detection is an important module in visual simultaneous localization and mapping (SLAM), which associates current camera observation with previously visited places. Loop closures correct drifts in trajectory estimation to build a globally consistent map. However, a false loop closure can be fatal, so verification is required as an additional step to ensure robustness by rejecting the false positive loops. Geometric verification has been a well-acknowledged solution that leverages spatial clues provided by local feature matching to find true positives. Existing feature matching methods focus on homography and pose estimation in long-term visual localization, lacking references for geometric verification. To fill the gap, this paper proposes a unified benchmark targeting geometric verification of loop closure detection under long-term conditional variations. Furthermore, we evaluate six representative local feature matching methods (handcrafted and learning-based) under the benchmark, with in-depth analysis for limitations and future directions.

### Run-time Analysis
We measure the runtime of six methods listed in Table I on NVIDIA GeForce RTX 3090 GPU and Intel i7-13700K CPU over 10K runs. The results are shown in figure below as inference time over performance, i.e., max recall @100 precision. We can conclude that the runtime of six local feature matching methods is at a millisecond level on a modern GPU. The choice can be made based on the trade-off between time efficiency and performance.
<p align="center">
<img src="./assets/figs/inference_time_vs_MR-crop.png" width="300" alt="Description">
</p>



## News
- :rocket: Releasing the visualization of [image matching](./assets/appendix.pdf) results. ([google drive](https://drive.google.com/file/d/1145hQb812E0HaPGekdpD04bEbjuej4Lx/view?usp=drive_link))

- :rocket: :rocket: Releasing the benchmark (easy)! Checkout the image pairs from `dataset/release/pairs` and images from [google drive](https://drive.google.com/drive/folders/1E8m353fi3hv-gaytJuRPLhFeNLPWTak6?usp=sharing) 
- :star: Benchmark usage is coming soon!

## Release Timeline
- [ ] Appendix for visualization
  - [x] Visualization of image matches
  - [ ] Visualization of inliers' distirbution
- [ ] Release benchmark sequences.
  - [x] Benchmark-easy 
    - [x] Day
    - [x] Weather
    - [x] Night-easy
    - [x] Season-easy
  - [ ] Benchmark-hard
- [ ] Release Local feature extraction and matching implementation
- [ ] Release evaluation tools
- [ ] Release data analysis tools
- [ ] Expansion to other verification methods
  - [ ] Dopplergangers
  - [ ] Semantics
  - [ ] Keypoint topology
- [ ] Release sequence version of benchmark


## Installation
We use part of the HLoc code for feature extraction and matching.  
```bash
git clone && cd GV-Bench
git submodule init
git submodule update
cd third_party/Hierarchival-Localization
python -m pip install -e .
```

## Usage
- Download the dataset sequences from [google drive](https://drive.google.com/file/d/1145hQb812E0HaPGekdpD04bEbjuej4Lx/view?usp=drive_link) and put it under the `dataset/` folder.
- Extract and match feature using hloc (Take SuperPoint and SuperGlue as an example).
  - Extract features: SIFT, SuperPoint, and DISK
    ```bash
    python hloc_utils.py --extraction --image_path /path/to image/ --output_path /path/to/output
    python hloc_utils.py --extraction --image_path dataset/images/ --output_path dataset/output/features/
    ```
  - Match features: SIFT-NN, SIFT-LightGlue, SuperPoint-NN, DISK-NN, SuperPoint-SuperGlue, SuperPoint-LightGlue, DISK-LightGlue, LoFTR
    ```bash
    python hloc_utils.py --matching 
    
    
    
    ```
  - Extract SuperPoint features on images.
    ```bash
    python gvbench_utils.py --image_path /path/to/image/folder/ --output_path /path/to/image/foler/ --feature superpoint_max #for example 
    ```
  - Match Superpoint features on image pairs
    - We prepare pairs file for matching under `pairs/` foler.
    - e.g. `day.txt` means single image pairs and `day_5.txt` means sequence image pairs.

<!-- ## Usage
### Installation
- Install `conda`
  
```bash
git clone --recursive
cd GV-BENCH/
conda create --name=gvbench python=3.8
cd third_party/Hierarchical-Localization/
python -m pip install -e .
``` -->


## Acknowledgement
- This work builds upon [hloc](https://github.com/cvg/Hierarchical-Localization), thanks for their amazing work.
- Contact: `jingwen.yu@connect.ust.hk`

