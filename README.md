# Validation of Visual Place Recognition (Loop Closure Detection)




# An Evaluation of Local Feature Matching for Geometrical Verification of Loop Closure Detection

## Intro
This repo contains the code of evaluation and dataset processing. [Hloc](https://github.com/cvg/Hierarchical-Localization) is our primary tool for extracting local features and matching features.

The 

## Usage

### Installation
- Install `conda`
  
```bash
git clone
cd GV-BENCH/
conda create --name=gvbench python=3.8
cd third_party/Hierarchical-Localization/
python -m pip install -e .
```

## TODO

### Dataset
- Oxford RobotCar
  - [x] Crop
  - [ ] Feature Extraction & Matching
  - [ ] Viewpoint Quantification
    - [ ] Find out the Maximum Viewpoint Variation
      - [ ] 5m
      - [ ] 10m
- RobotCar Seasons
- Mapillary Street Level Dataset

### Feature Extraction & Matching
#### Extraction
- Robotcar RCV
  - [ ] SIFT
  - [ ] ORB
  - [ ] SuperPoint

### Dataset Processing
#### Oxford RobotCar
- [x] Image Crop (To eliminate the effect caused by the vehicle hood.)
- [ ] Viewpoint Quantification of two images 

### Evaluation



