# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

### [[Arxiv]](https://arxiv.org/abs/2407.11736) [[Project Page]](https://jarvisyjw.github.io/GV-Bench/) [[Intro in Chinese]](https://mp.weixin.qq.com/s/edUw7vLep0zmve0Uj3IzkQ)
<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<!-- <p align="center">
<img src="./assets/figs/radar-final-iros.png" width="600" alt="Description">
</p>

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
- :star::star::star: Add support for [Image-matching-models](https://github.com/alexstoken/image-matching-models), which makes it easy for evaluating different image matching models on GV-Bench. Thanks for their excellent work!!!

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
Please follow the installation of [image-matching-models](https://github.com/alexstoken/image-matching-models)
```bash

git clone --recursive https://github.com/jarvisyjw/GV-Bench.git
```

<!-- We use part of the HLoc code for feature extraction and matching.  
```bash
git clone && cd GV-Bench
git submodule init
git submodule update
cd third_party/Hierarchival-Localization
git checkout gvbench # this is a customized fork version
python -m pip install -e .
``` -->

## Replicate Results in Exps

`Note that we use fundamental matrix estimation for outlier reject and use default image resolution of 1024*640, which is different from the image-matching-model's processing. Therefore if you replicate the experiments using image-matching-models, you may get different results.`

We provide the [output results](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jyubt_connect_ust_hk/EkflAPp79spCviRK5EkSGVABrGncg-TfNV5I3ThXxzopLg?e=tu91Xn) with the format shown below. You can use these results directly.

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

- Evaluation


## TODO
- [x] Enabling using customized local features for geometric verification (GV).

## Acknowledgement
- This work thanks [hloc](https://github.com/cvg/Hierarchical-Localization), [image-matching-models](https://github.com/alexstoken/image-matching-models) for their amazing works.
- Contact: `jingwen.yu@connect.ust.hk`

