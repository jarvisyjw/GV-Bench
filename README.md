# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<p align="center">
<img src="./assets/figs/radar-images.png" width="500" alt="Description">
</p>

## Intro
This repo contains the implementation of GV-Bench, aiming at providing a fair and accessible benchmark for geometric verification.

## Timeline
- [ ] Release benchmark sequences.
  - [ ] Day
  - [ ] Night-easy
  - [ ] Night-hard
  - [ ] Season-easy
  - [ ] Season-hard
- [ ] Release Local feature extraction and matching implementation
- [ ] Release evaluation tools
- [ ] Release data analysis tools


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
- This work builds upon `hloc`, thanks for their amazing work.

