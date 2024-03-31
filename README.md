# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<p align="center">
<img src="./assets/figs/radar-images.png" width="600" alt="Description">
</p>

<!-- ## Visualization of Image Matching -->

## Introduction
(Under construction, full codes and results comming soon!) Feel free to dorp me an email or leave an issue!

This repo contains the implementation of GV-Bench, aiming at providing a fair and accessible benchmark for geometric verification. We employ three datasets Oxford Robotcar, Nordland, and UAcampus, containing appearance changes over long period. 

## News
- Releasing the visualization of [image matching](./assets/appendix.pdf) results. ([google drive link](https://drive.google.com/file/d/1145hQb812E0HaPGekdpD04bEbjuej4Lx/view?usp=drive_link))

## Release Timeline
- [ ] Appendix for visualization
  - [x] Visualization of image matches
  - [ ] Visualization of inliers' distirbution
- [ ] Release benchmark sequences.
  - [ ] Day
  - [ ] Night-easy
  - [ ] Night-hard
  - [ ] Season-easy
  - [ ] Season-hard
- [ ] Release Local feature extraction and matching implementation
- [ ] Release evaluation tools
- [ ] Release data analysis tools
- [ ] Expansion to other verification methods
  - [ ] Dopplergangers
  - [ ] Semantics
  - [ ] Keypoint topology
- [ ] Release sequence version of benchmark


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

