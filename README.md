# GV-Bench

> <b>GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection</b> <br>
> [Jingwen Yu](https://jingwenyust.github.io/), [Hanjing Ye](https://medlartea.github.io/), [Jianhao Jiao](https://gogojjh.github.io/), [Ping Tan](https://facultyprofiles.hkust.edu.hk/profiles.php?profile=ping-tan-pingtan), [Hong Zhang](https://faculty.sustech.edu.cn/?tagid=zhangh33&iscss=1&snapid=1&orderby=date&go=2&lang=en)<br>
> 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) [arXiv](https://arxiv.org/abs/2407.11736), [IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10801481), [Project Page](https://jarvisyjw.github.io/GV-Bench/), [Blog Post (in Chinese)](https://mp.weixin.qq.com/s/edUw7vLep0zmve0Uj3IzkQ), [Video (Bilibili)](https://www.bilibili.com/video/BV1WD23YhEZw/?share_source=copy_web&vd_source=4db6a86d3347fa85196b3e77a6092d1a)

<p align="center">
<img src=".asset/radar-webpage.png" width="600" alt="Description">
</p>

## Abstract
In short, GV-Bench provides a benchmark for evaluating different local feature matching methods on geometric verification (GV), which is crucial for vision-based localization and mapping systems (e.g., Visual SLAM, Structure-from-Motion, Visual Localization).

## News
- :star: Add support for [Image-matching-models](https://github.com/alexstoken/image-matching-models), thanks for their excellent work! :smile:
- :tada: [Chinese intro](https://mp.weixin.qq.com/s/edUw7vLep0zmve0Uj3IzkQ) on wechat offical account.
- :star: Paper is release on [arxiv](https://arxiv.org/abs/2407.11736).
- :tada: The paper is accepted by IROS 2024!
- :rocket: Releasing the visualization of [image matching]([./assets/appendix.pdf](https://drive.google.com/file/d/1145hQb812E0HaPGekdpD04bEbjuej4Lx/view?usp=drive_link)) results.
- :rocket: Releasing the benchmark!

## Installation
Please follow the installation of [image-matching-models](https://github.com/alexstoken/image-matching-models)
```bash
# clone the repo
git clone --recursive https://github.com/jarvisyjw/GV-Bench.git
cd third_party/image-matching-models
git submodule init
git submodule update
conda create -n gvbench python=3.11
python -m pip install -U prettytable
# Then follow the installation of image-matching-models
```

## Evaluation
### Data
1. Get the GV-Bench sequences from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jyubt_connect_ust_hk/EkflAPp79spCviRK5EkSGVABrGncg-TfNV5I3ThXxzopLg?e=DdwCAL).
2. Unzip and organize the dataset folder like following:
  
  ```bash
    |-- gt
    |   |-- day.txt
    |   |-- night.txt
    |   |-- nordland.txt
    |   |-- season.txt
    |   |-- uacampus.txt
    |   |-- weather.txt
    |-- images
    |   |-- day0
    |   |-- day1
    |   |-- night0
    |   |...
  ```
### Benchmark
Now, we support using [image-matching-models](https://github.com/alexstoken/image-matching-models) directly,
which enables many more state-of-the-art matching methods for Geometric Verification (GV). 
The example usage:

```bash
python main.py config/day.yaml
```

In the configs, please specify the `data directory`, `sequence info` (which is provided with the repo in dataset/gt folder), `image_size` and the `matcher` you want to use.
The evaluation supports runing multiple matchers in a single run. However, our sequences provides a rather large number of pairs, so the evaluation might takes time.

If you want to replicate the paper's result of IROS2024, please refer to [this](./.asset/replicate.md).

### Using Customized Image Matcher
- We recommend contributing or building your image matcher as the standard of [image-matching-models](https://github.com/alexstoken/image-matching-models). 
- Example usage of bench loader and evaluator
  ```python
  # bench sequence
  gvbench_seq = ImagePairDataset(config.data, transform=None) # load images
  labels = gvbench_seq.label # load labels
  MODEL = YOUR_MODEL(max_num_keypoints=2048) # your image matching model
  # if your method is two-stage image matching
  # i.e. Step 1: Keypoints Extraction
  #      Step 2: Keypoints matching
  # We recommend you set a max_num_keypoints to 2048
  # to be consistent with the image-matching-models default setup.
  scores = []
  for data in gvbench_seq:
    img0 = load_image(data['img0'], resize)
    img1 = load_image(data['img1'], resize)
    inliers = MODEL(img0, img1)
    scores.append(inliers)
  # normalize
  scores_norm = (scores - np.min(scores)) / (np.max(scores)- np.min(scores))
  mAP, MaxR = eval(scores, labels)
  ```

## Acknowledgement
- This work thanks [hloc](https://github.com/cvg/Hierarchical-Localization), [image-matching-models](https://github.com/alexstoken/image-matching-models) for their amazing works.
- Contact: `jingwen.yu@connect.ust.hk`

## Citation
```
@inproceedings{yu2024gv,
  title={GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection},
  author={Yu, Jingwen and Ye, Hanjing and Jiao, Jianhao and Tan, Ping and Zhang, Hong},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={7922--7928},
  year={2024},
  organization={IEEE}
}
```