# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<p align="center">
<img src="./assets/figs/radar-images.png" width="600" alt="Description">
</p>

## Introduction
(Under construction, full codes and results comming soon!)

This repo contains the implementation of GV-Bench, aiming at providing a fair and accessible benchmark for geometric verification. We employ three datasets Oxford Robotcar, Nordland, and UAcampus, containing appearance changes over long period. 

## News
- Releasing the visualization of [image matching](./assets/appendix.pdf) results.

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


## Usage
### Installation
- Install `conda`
  
```bash
git clone --recursive
cd GV-BENCH/
conda create --name=gvbench python=3.8
cd third_party/Hierarchical-Localization/
python -m pip install -e .
```


## Acknowledgement
- This work builds upon [hloc](https://github.com/cvg/Hierarchical-Localization), thanks for their amazing work.

