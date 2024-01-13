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
  - [ ] Post-processing the existing lists
    - [ ] 

### Evaluation







# Sequential Loop Closure Verification Dataset

This dataset is derived from [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/datasets/) contains three appearance-changing (Day-Night, Seasonal, Temporal) repetitive routes (SunCloud, Night, Autumn).

```bash
SunCloud: 2014-11-18-13-20-12/stereo/centre/
Night: 2014-12-16-18-44-24/stereo/centre/ # of images 32585
Autumn: 2014-12-09-13-21-02/stereo/centre/ 
```

## Dataset
```bash
*_val: Complete dataset with images and gps info
*_mini_val: Minimal dataset with only a portion of images
```

## Ground Truth
```bash
GT format
      query, reference, label (1/0)

Day & Night (with dynamic objects)
      robotcar_qAutumn_dbNight.txt

Viewpoint Changes (with dynamic objects)
      robotcar_qAutumn_dbSuncloud.txt

```

### Quantized ViewPoint Level:
- [x] RobotCar Autumn & Night Sequence
- [x] RobotCar Autumn & Suncloud Sequence
- [ ] RobotCar ....
```shell
# For each data sequence, there will be 5 extra gt sequences
robotcar_qAutumn_db(Night/Suncloud)_dist_{N}.txt # N=0:4
# view angle in degrees and dist in meters
dist_0: view angle < 30 && dist < 25
dist_1: view angle < 10 && dist < 5
dist_2: view angle < 5  && dist < 0.5
dist_4: view angle < 2  && dist < 0.25
dist_5: complement set of dist_0
```



