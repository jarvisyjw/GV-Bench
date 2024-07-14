#%%
from tqdm import tqdm
file_path = 'dataset/gt/netvlad_pairs_gt2.txt'
file_path_new = 'dataset/gt/season-hard.txt'
fout = open(file_path_new, 'w')

with open(file_path, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines):
      q, r, l = line.strip('\n').split(' ')
      print(f'q: {q}, r: {r}')
      new_q = q.replace('Live', 'season1')
      new_r = r.replace('Reference', 'season2')
      print(f'new_q: {new_q}, new_r: {new_r}')
      fout.write(f'{new_q} {new_r} {l}\n')

#%%
import shutil
file_path = 'dataset/release/gt/weather.txt'

with open(file_path, 'r') as f:
    lines = f.readlines()

for line in lines:
      q, r, l = line.strip('\n').split(' ')
      # source_file = f'dataset/images/{q}.jpg'
      shutil.copy(f'dataset/images/{r}', f'dataset/release/images/{r}')
      
# %%
file_path = 'dataset/gt/season.txt'
file_path_new = 'dataset/release/gt/season.txt'
fout = open(file_path_new, 'w')

with open(file_path, 'r') as f:
    lines = f.readlines()
print(len(lines))
 
for line in lines:
      q, r, l = line.strip('\n').split(', ')
      # print(f'q: {q}, r: {r}, l: {l}')
      new_q = q.replace('Autumn_mini_val', 'day0')
      new_r = r.replace('Snow_mini_val', 'season0')
      # print(f'new_q: {new_q}, new_r: {new_r}')
      fout.write(f'{new_q} {new_r} {int(l)}\n')
fout.close()
# %%
'''
    Check if the images are valid
'''
import os

file_path = 'dataset/release/pairs/weather.txt'

with open(file_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    q, r = line.strip('\n').split(' ')
    if not os.path.exists(f'dataset/release/images/{r}'):
        print(f'{r} does not exist')
    # print(f'q: {q}, r: {r}')
    # # source_file = f'dataset/images/{q}.jpg'
    # shutil.copy(f'dataset/images/{r}', f'dataset/release/images/{r}')

# %%
# generate sequential images
# load pairs
# load day-night pairs
from utils import parse_pairs, logger
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# %%
def find_closest(lst, x):
    """
    在有序列表 lst 中找到与给定值 x 大小最接近的元素。
    """
    n = len(lst)
    if x <= lst[0]:
        return lst[0]
    elif x >= lst[n-1]:
        return lst[n-1]
    else:
        # 二分查找
        low, high = 0, n-1
        while low <= high:
            mid = (low + high) // 2
            if lst[mid] == x:
                return lst[mid]
            elif lst[mid] < x:
                low = mid + 1
            else:
                high = mid - 1

        # 最后返回距离 x 最近的元素
        if lst[high] - x < x - lst[low]:
            return lst[high]
        else:
            return lst[low]
        
        
# %% interpolate_ins_poses for all images
from pathlib import Path
from interpolate_poses import interpolate_ins_poses
from transform import se3_to_components
import pandas as pd
from tqdm import tqdm
import numpy as np

gps_root = 'dataset/gps'
gps_day = Path(gps_root, 'day0')
gps_night = Path(gps_root, 'night0')

image_path = Path('dataset/robotcar/Autumn_val/stereo/centre')
image_t = [image.name.strip('.jpg') for image in image_path.iterdir()]

ins_frame = Path(gps_day, 'ins.csv')
ins_pd = pd.read_csv(ins_frame)
ts = ins_pd['timestamp'].tolist()

dataframe = []

for t in tqdm(image_t[:10]):
    t_closest = min(ts, key=lambda x:abs(x-int(t)))
    t_closest_xyz = ins_pd[ins_pd['timestamp'] == t_closest].iloc[0].values[5:7]
    t_closest_rpy = ins_pd[ins_pd['timestamp'] == t_closest].iloc[0].values[-3:]
    t_closest_xyzrpy = np.concatenate((t_closest_xyz, t_closest_rpy))
    print(f'original: {t_closest_xyzrpy}')
    _, t_pose = interpolate_ins_poses(str(ins_frame), [int(t)], t_closest)
    t_xyzrpy = se3_to_components(t_pose[0])
    np.set_printoptions(precision=12, suppress=True)
    print(f'interlolated: {t_xyzrpy}')
    dataframe.append([t, t_closest, t_closest_xyzrpy, t_xyzrpy])

df = pd.DataFrame(dataframe, columns=['timestamp', 'timestamp_closest', 'closest_xyzrpy', 'xyzrpy'])
df.to_csv('dataset/gps/day0/interpolate_poses.csv', index=True)

# %%
from utils import generate_sequence
from pathlib import Path

image_path = Path('dataset/robotcar/Autumn_val/stereo/centre')
image_t = [int(image.name.strip('.jpg')) for image in image_path.iterdir()]
seq = generate_sequence('dataset/gps/day0/interpolate_poses.csv', image_t, 5)




# %%
from interpolate_poses import interpolate_ins_poses

loader = parse_pairs('dataset/release/pairs/night.txt', allow_label=False)
gps_root = 'dataset/gps'
gps_day = Path(gps_root, 'day0')
gps_night = Path(gps_root, 'night0')

interpolate_ins_poses(gps_day, [1418132441421073], 1418132441421073)


def get_seq(gps_path: str, t: str):
    '''
    
    query_dir: /path/to/sequence/
        /mnt/DATA_JW/dataset/VPR/robotcar/2014-12-09-13-21-02
    ref_dir: /path/to/sequence/
        /mnt/DATA_JW/dataset/VPR/robotcar/2014-12-09-13-21-02
    
    query_t: timestamp
        
    ref_t: timestamp
    '''
    ins_frame = Path(gps_path, 'ins.csv')
    ins_pd = pd.read_csv(ins_frame)
    ts = ins_pd['timestamp'].tolist()
    t = min(ts, key=lambda x:abs(x-int(t)))
    
    # t = find_closest(ts, int(t))
    return t

# %%

t = find_closest('dataset/gps/day0','1418132441421073')

print(t)


# %%
image_path = Path('dataset/robotcar/Autumn_val/stereo/centre')
t_day_all = [image.name.strip('.jpg') for image in image_path.iterdir()]
# %%
print(t_day_all)
# %%


# 1418132440421211
# 1418132440409306 : find_closest
# 1418132440429309

# 1418132441421073
# 1418132441409353 : find_closest
# 1418132441429356

# %%
from utils import interpolate_poses
from pathlib import Path
image_path = Path('dataset/robotcar/Autumn_val/stereo/centre')
interpolate_poses(image_path, 'dataset/gps/day0/ins.csv', 'dataset/gps/day0/interpolate_poses.csv')
# %%
from tqdm import tqdm
file_path = 'dataset/gt/day.txt'
output_path = 'dataset/pairs/day1.txt'
fout = open(output_path, 'w')

with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        q, r, l = line.strip('\n').split(' ')
        q = q.split('/')[-1]
        # q = f'day0/{q}'
        # r = f'day1/{r}'
        fout.write(f'{q}\n')
fout.close()
f.close()
# %%
from pathlib import Path
from tqdm import tqdm

image_path = Path('dataset/GV-Bench/images/night0')
image_t = [image.stem for image in image_path.iterdir()]

with open('dataset/pairs/night0.txt', 'w') as f:
    for t in tqdm(image_t):
        f.write(f'{t}\n')
        

# %%
import pandas as pd
from tqdm import tqdm

gt_path = 'dataset/gt/weather.txt'
images_list = 'dataset/sequences/day0/seq_5.csv'
image_list_ref = 'dataset/sequences/weather0/seq_5.csv'
output_path = 'dataset/sequences/weather.txt'
fout = open(output_path, 'w')
with open(gt_path, 'r') as f:
    lines = f.readlines()

csv_file = pd.read_csv(images_list)
q_target_t = csv_file['timestamp'].tolist()
r_target_t = pd.read_csv(image_list_ref)['timestamp'].tolist()
# print(target_t)

for line in tqdm(lines):
    q, r, l = line.strip('\n').split(' ')
    r_t = int(r.strip('.jpg').split('/')[-1])
    q_t = int(q.strip('.jpg').split('/')[-1])
    if q_t in q_target_t and r_t in r_target_t:      
        fout.write(f'{q} {r} {l}\n')
fout.close()

# %%
root_dir = 'dataset/robotcar/Autumn_val/stereo/centre'
target_dir = 'dataset/GV-Bench/release/images_seq/day0'
