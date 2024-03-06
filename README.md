# GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection

<!-- ![GV-Bench](./assets/figs/radar-chart.png ) -->
<p align="center">
<img src="./assets/figs/radar-chart.png" width="500" alt="Description">
</p>

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
      - [ ] Max view:
      - [ ] Max distance:
      - [ ] Max view & distance
    - [x] Split different Range
      - [x] 0.25m, $2^o$
      - [x] 0.5m, $5^o$
      - [x] 5m, $10^o$
      - [x] 25m, $30^o$ 
- RobotCar Seasons
- Mapillary Street Level Dataset

### Feature Extraction & Matching
#### Extraction & Matching
- Robotcar GV
  - [ ] SIFT -> NN, (TODO: LightGlue)
    - [x] Extraction
    - [x] Matching
  - [ ] ORB -> (TODO: LightGlue)
    - [ ] NN
  - [ ] SuperPoint -> 
    - [x] NN 
    - [ ] LightGlue
    - [x] SuperGlue
  - [x] LoFTR
- TODO
  - [ ] Doppelgangers
  - [ ] GMS/COTR/Huang Xinghong's Method

### Dataset Processing
#### Oxford RobotCar
- [x] Image Crop (To eliminate the effect caused by the vehicle hood.) `in each {dataset name}_mini_val` folder
- [x] Viewpoint Quantification of two images
  - [x] Post-processing the existing lists
    - [x] dist_{0:4}
  
<!-- #### Oxford RobotCar Seasons V2 -->

### Evaluation
- [ ] SIFT + NN
- [ ] ORB + NN
- [ ] SuperPoint + NN
- [ ] SuperPoint + SuperGlue
- [ ] Disk + SuperGlue
- [ ] LoFTR


# Loop Closure Verification Dataset 
(Based on Oxford RobotCar)

This dataset is derived from [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/datasets/) contains three appearance-changing (Day-Night, Seasonal, Temporal) repetitive routes (SunCloud, Night, Autumn).

```bash
# Day Night Changes
Day to Day: qAutumn_dbSunCloud
Day to Night: qAutumn_dbNight

# Seasonal Changes
Summer to Winter: qSummer_dbWinter
Autumn to Winter: qAutumn_dbSnow

# Weather Changes
Night to Rain: qNight_dbRain
```


```bash
Day to Day: SunCloud 2014-11-18-13-20-12/stereo/centre/


SunCloud: 2014-11-18-13-20-12/stereo/centre/
Night: 2014-12-16-18-44-24/stereo/centre/ # of images 32585
Autumn: 2014-12-09-13-21-02/stereo/centre/

Snow-winter: 2015-02-03-08-45-10/stereo/centre/
Night-Rain: 2014-12-17-18-18-43/

Overcast-Summer: 2015-05-22-11-14-30


Overcast-Sun: 2015-11-13-10-28-08
Overcast: 2014-12-02-15-30-08

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



