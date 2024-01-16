import numpy as np
import os.path as osp
import os
import shutil
import cv2
from tqdm import tqdm
import pandas as pd
import math
import random
from pathlib import Path
from multiprocessing import Process, Queue
import multiprocessing, threading


from gvbench.utils import gt_loader, load_gt, parse_pairs, write_pairs
from gvbench import logger

os.environ['OPENCV_PYTHON_HIDE_WINDOW'] = 'true'
np.random.seed(3)
random.seed(3)

def read_and_store(fname, fname_mini, q_dir_name, q_mini_dir_name, ref_dir_name, ref_mini_dir_name):
    print("------------------------------------------")
    print("Construct: {}".format(fname_mini))
    print("------------------------------------------")
    mini_query_dir = q_mini_dir_name
    if not osp.exists(mini_query_dir):
        os.mkdir(mini_query_dir)
    mini_ref_dir = ref_mini_dir_name
    if not osp.exists(mini_ref_dir):
        os.mkdir(mini_ref_dir)
    
    f_mini_p = open(fname_mini, 'w')
    with open(fname, 'r') as f_p:
        for i, line in enumerate(f_p):
            f_str = line.strip("\n").split('; ')
            q_name = osp.join(q_dir_name, "stereo/centre", osp.basename(f_str[0]))
            ref_str = f_str[1]
            ref_names = [osp.join(ref_dir_name, "stereo/centre", osp.basename(ref_name.strip('\'')))  for ref_name in ref_str.strip('[]').split(', ')]
            gt_str = f_str[2]
            gt_names = [osp.join(ref_dir_name, "stereo/centre", osp.basename(gt_name.strip('\''))) for gt_name in gt_str.strip('[]').split(', ')]

            q_mini_name = osp.join(mini_query_dir, osp.basename(q_name))
            if not osp.exists(q_mini_name):
                shutil.copy(q_name, mini_query_dir)
            for ref_name in ref_names:
                ref_mini_name = osp.join(mini_ref_dir, osp.basename(ref_name))
                if not osp.exists(ref_mini_name):
                    shutil.copy(ref_name, mini_ref_dir)
                if len(gt_names) == 0:
                    continue
                elif ref_name in gt_names:
                    f_mini_p.write("{}, {}, {}\n".format(q_mini_name, ref_mini_name, 1))
                else:
                    f_mini_p.write("{}, {}, {}\n".format(q_mini_name, ref_mini_name, 0))
            if len(gt_names) == 0:
                for gt_name in gt_names:
                    if not osp.exists(osp.join(mini_ref_dir, osp.basename(gt_name))):
                        shutil.copy(gt_name, mini_ref_dir)
    f_mini_p.close()

def construct_dataset(fname_mini, fname_easy, fname_difficult, nums=20000):
    print("------------------------------------------")
    print("Construct: {} and {}".format(fname_easy, fname_difficult))
    print("------------------------------------------")
    fname_easy = open(fname_easy, 'w')
    fname_difficult = open(fname_difficult, 'w')
    easy_pos_nums = 0
    easy_neg_nums = 0
    diff_pos_nums = 0
    diff_neg_nums = 0
    with open(fname_mini, 'r') as f_p:
        for i, line in enumerate(f_p):
            f_str = line.split(', ')
            q_img_fname = f_str[0]
            ref_img_fname = f_str[1]
            is_loop = int(f_str[2])

            eval_loop = loop_verification(q_img_fname, ref_img_fname, (640,480))
            if eval_loop == is_loop:
                if is_loop == 1 and easy_pos_nums < nums:
                    easy_pos_nums += 1
                    fname_easy.write("{}, {}, {}\n".format(q_img_fname, ref_img_fname, is_loop))
                elif is_loop == 0 and easy_neg_nums < nums:
                    easy_neg_nums += 1
                    fname_easy.write("{}, {}, {}\n".format(q_img_fname, ref_img_fname, is_loop))
            else:
                if is_loop == 1 and diff_pos_nums < nums:
                    diff_pos_nums += 1
                    fname_difficult.write("{}, {}, {}\n".format(q_img_fname, ref_img_fname, is_loop))
                elif is_loop == 0 and diff_neg_nums < nums:
                    diff_neg_nums += 1
                    fname_difficult.write("{}, {}, {}\n".format(q_img_fname, ref_img_fname, is_loop))
            print("{}, {} {} {} {}".format(i, easy_pos_nums, easy_neg_nums, diff_pos_nums, diff_neg_nums))
            if easy_pos_nums > nums and easy_neg_nums > nums and diff_pos_nums > nums and diff_neg_nums > nums:
                break
    fname_easy.close()
    fname_difficult.close()



def calculate_distance(north1, east1, north2, east2):
    # 计算距离
    dx = east2 - east1
    dy = north2 - north1
    distance = math.sqrt(dx**2 + dy**2)

    return distance

def calculate_view(yaw1, yaw2):
    return abs(yaw1-yaw2)/3.1415926*180

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

def store_mini_dataset(fname, final_fname, q_original_dir_name, ref_original_dir_name, q_mini_dir_name, ref_mini_dir_name, is_large_view=False, nums=400, large_dis_threshold=15, small_dis_threshold=5, view_threshold=70):
    """Only extract certain numbers
    
    """
    print("------------------------------------------")
    print("Construct: {}".format(final_fname))
    print("------------------------------------------")

    query_ins_fname = osp.join(q_original_dir_name, "gps/ins.csv")
    query_df = pd.read_csv(query_ins_fname)
    query_df_timestamps = query_df["timestamp"].tolist()
    ref_ins_fname = osp.join(ref_original_dir_name, "gps/ins.csv")
    ref_df = pd.read_csv(ref_ins_fname)
    ref_df_timestamps = ref_df["timestamp"].tolist()


    mini_query_dir = q_mini_dir_name
    if not osp.exists(mini_query_dir):
        os.mkdir(mini_query_dir)
    mini_ref_dir = ref_mini_dir_name
    if not osp.exists(mini_ref_dir):
        os.mkdir(mini_ref_dir)
    
    fo_p = open(final_fname, 'w')
    q_names = []
    ref_names = []
    labels = []
    with open(fname, 'r') as f_p:
        for i, line in enumerate(f_p):
            f_str = line.split(', ')
            q_names.append(f_str[0])
            ref_names.append(f_str[1])
            labels.append(int(f_str[2]))

    q_pos_fnames = []
    ref_pos_fnames = []
    pos_labels = []
    q_neg_fnames = []
    ref_neg_fnames = []
    neg_labels = []
    for i, label in enumerate(labels):
        if label == 1:
            q_pos_fnames.append(q_names[i])
            ref_pos_fnames.append(ref_names[i])
            pos_labels.append(1)
        else:
            q_neg_fnames.append(q_names[i])
            ref_neg_fnames.append(ref_names[i])
            neg_labels.append(0)


    # pos_indexes = np.random.choice(range(len(pos_labels)), size=nums, replace=False)
    # pos_indexes = range(len(pos_labels))

    # print(pos_indexes)
    pos_candidates = []
    for index in tqdm(range(len(pos_labels))):
        q_mini_name = osp.join(mini_query_dir, osp.basename(q_pos_fnames[index]))
        q_mini_t = int(osp.basename(q_pos_fnames[index]).strip(".jpg"))

        ref_mini_name = osp.join(mini_ref_dir, osp.basename(ref_pos_fnames[index]))
        ref_mini_t = int(osp.basename(ref_pos_fnames[index]).strip(".jpg"))

        
        closest_q_mini_t = find_closest(query_df_timestamps, q_mini_t)
        q_north, q_east, q_yaw = query_df[query_df["timestamp"]==closest_q_mini_t]["northing"].item(), query_df[query_df["timestamp"]==closest_q_mini_t]["easting"].item(), query_df[query_df["timestamp"]==closest_q_mini_t]["yaw"].item()
        # print(q_north, q_east)
        closest_ref_mini_t = find_closest(ref_df_timestamps, ref_mini_t)
        ref_north, ref_east, ref_yaw = ref_df[ref_df["timestamp"]==closest_ref_mini_t]["northing"].item(), ref_df[ref_df["timestamp"]==closest_ref_mini_t]["easting"].item(), ref_df[ref_df["timestamp"]==closest_ref_mini_t]["yaw"].item()
        dis = calculate_distance(q_north, q_east, ref_north, ref_east)
        view = calculate_view(q_yaw, ref_yaw)
        # print(dis)

        if (is_large_view and dis > large_dis_threshold and view < view_threshold) or (not is_large_view and dis < small_dis_threshold and view < view_threshold):
            pos_candidates.append((q_mini_name, q_pos_fnames[index], ref_mini_name, ref_pos_fnames[index]))

    pos_indexes = np.random.choice(range(len(pos_candidates)), size=nums, replace=False)
    for index in pos_indexes:
        q_mini_name, q_pos_fname, ref_mini_name, ref_pos_fname = pos_candidates[index]
        if not osp.exists(q_mini_name):
            shutil.copy(q_pos_fname, mini_query_dir)
        if not osp.exists(ref_mini_name):
            shutil.copy(ref_pos_fname, mini_ref_dir)
        fo_p.write("{}, {}, {}\n".format(q_mini_name, ref_mini_name, pos_labels[index]))
    
    neg_indexes = np.random.choice(range(len(neg_labels)), size=nums, replace=False)
    # neg_indexes = range(len(neg_labels))

    for index in neg_indexes:
        q_mini_name = osp.join(mini_query_dir, osp.basename(q_neg_fnames[index]))
        ref_mini_name = osp.join(mini_ref_dir, osp.basename(ref_neg_fnames[index]))
        if not osp.exists(q_mini_name):
            shutil.copy(q_neg_fnames[index], mini_query_dir)
        if not osp.exists(ref_mini_name):
            shutil.copy(ref_neg_fnames[index], mini_ref_dir)
        fo_p.write("{}, {}, {}\n".format(q_mini_name, ref_mini_name, neg_labels[index]))
    fo_p.close()

def loop_verification(img_fname_1, img_fname_2, new_size=None, is_vis=False):
    # 读取两幅图像
    img1 = cv2.imread(img_fname_1)
    if new_size is not None:
        img1 = cv2.resize(img1, new_size)
    img2 = cv2.imread(img_fname_2)
    if new_size is not None:
        img2 = cv2.resize(img2, new_size)

    # 提取关键点和描述符
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    # 进行特征匹配并进行几何验证
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # matches = bf.match(des1, des2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    # good_matches = []
    # for m in matches:
        # if m.distance < 50: # 设定一个阈值来排除不合理的匹配
            # good_matches.append(m)
    if len(good_matches) > 30: # 判断是否存在闭环
        if is_vis:
            print('Found loop closure!')
        else:
            return 1
    else:
        if is_vis:
            print('No loop closure found.')
        else:
            return 0

    # 可视化匹配结果和验证结果
    if is_vis:
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def glance_img(fname, new_size=(640,480)):
    with open(fname, 'r') as f_p:
        for i, line in enumerate(f_p):
            f_str = line.split(', ')
            q_name = f_str[0]
            ref_name = f_str[1]
            label = int(f_str[2])
            if label == 1:
                loop_verification(q_name, ref_name, new_size=new_size, is_vis=True)

def concatenate_txt(fname1, fname2, final_fname):
    lis = []
    with open(fname1, 'r') as f_p:
        for i, line in enumerate(f_p):
            f_str = line.strip().split(', ')
            f_str = f_str[0] + ", " + f_str[1]
            lis.append(f_str)
    with open(fname2, 'r') as f_p:
        for i, line in enumerate(f_p):
            f_str = line.strip().split(', ')
            f_str = f_str[0] + ", " + f_str[1]
            lis.append(f_str)
    random.shuffle(lis)
    f = open(final_fname, "w")
    for line in lis:
        f.write(line + "\n")
    f.close()


def cal_two_frame_dis_yaw(query_dir, ref_dir, query_t, ref_t):
    query_ins_fname = osp.join(query_dir, "gps/ins.csv")
    query_df = pd.read_csv(query_ins_fname)
    query_df_timestamps = query_df["timestamp"].tolist()
    ref_ins_fname = osp.join(ref_dir, "gps/ins.csv")
    ref_df = pd.read_csv(ref_ins_fname)
    ref_df_timestamps = ref_df["timestamp"].tolist()

    closest_q_mini_t = find_closest(query_df_timestamps, query_t)
    q_north, q_east, q_yaw = query_df[query_df["timestamp"]==closest_q_mini_t]["northing"].item(), query_df[query_df["timestamp"]==closest_q_mini_t]["easting"].item(), query_df[query_df["timestamp"]==closest_q_mini_t]["yaw"].item()
    # print(q_north, q_east)
    closest_ref_mini_t = find_closest(ref_df_timestamps, ref_t)
    ref_north, ref_east, ref_yaw = ref_df[ref_df["timestamp"]==closest_ref_mini_t]["northing"].item(), ref_df[ref_df["timestamp"]==closest_ref_mini_t]["easting"].item(), ref_df[ref_df["timestamp"]==closest_ref_mini_t]["yaw"].item()
    dis = calculate_distance(q_north, q_east, ref_north, ref_east)
    view = calculate_view(q_yaw, ref_yaw)
    dis = calculate_distance(q_north, q_east, ref_north, ref_east)
    return view, dis
    # print("Distance:", dis)
    # print("View:", view)


def gt_gen(queue, pairs_timestamp, query_dataset_dir, ref_dataset_dir):
    gt = []
    for pair_t in tqdm(pairs_timestamp, total = len(pairs_timestamp)):
        q_t, r_t = pair_t
        view, dis = cal_two_frame_dis_yaw(query_dataset_dir, ref_dataset_dir, int(q_t), int(r_t))
        # pair = (f"Autumn_mini_val/{q_t}.jpg", f"Night_mini_val/{r_t}.jpg", label_t)
        # logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
        if (view < 40 and dis < 25):
            pair = (f"Autumn_mini_val/{q_t}.jpg", f"Suncloud_mini_val/{r_t}.jpg", 1)
            logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
            gt.append(pair)
        else:
            pair = (f"Autumn_mini_val/{q_t}.jpg", f"Suncloud_mini_val/{r_t}.jpg", 0)
            logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
            gt.append(pair)
    queue.put(gt)
    
    
def gt_gen_multiprocess(pairs, query_dataset, ref_dataset, output_dir, num_process=8):
    query_dataset_dir = f"dataset/robotcar/{query_dataset}"
    ref_dataset_dir = f"dataset/robotcar/{ref_dataset}"
    pairs_loader = parse_pairs(pairs)
    pairs_timestamp = [(q.strip('.jpg').split('/')[-1], r.strip('.jpg').split('/')[-1]) for q, r in pairs_loader]
    
    gts = []
    pairs_timestamp = [pairs_timestamp[i::num_process] for i in range(num_process)]
    q = Queue()
    processes = []
    # rets = [[dist_1, dist_2, dist_3, dist_4, dist_null]]
    for i in range(num_process):
            p = Process(target=gt_gen, args=(q, pairs_timestamp[i], query_dataset_dir, ref_dataset_dir))
            processes.append(p)
            p.start()
    for p in processes:
        gts.extend(q.get())
    for p in processes:
        p.join()
        
    # import pdb; pdb.set_trace()
    # print(dist, len(dists))
    # concate_lists = concate_list(dists)
    return gts, len(gts)
    

def split_view(queue, pairs_timestamp, query_dataset_dir, ref_dataset_dir):
    # query_dataset_dir = f"dataset/robotcar/{query_dataset}"
    # ref_dataset_dir = f"dataset/robotcar/{ref_dataset}"
    # pairs_loader = parse_pairs(pairs)
    # pairs_timestamp = [(q.strip('.jpg').split('/')[-1], r.strip('.jpg').split('/')[-1], label) for q, r, label in pairs_loader]
    dist_1 = []
    dist_2 = []
    dist_3 = []
    dist_4 = []
    dist_null = []
    for pair_t in tqdm(pairs_timestamp, total = len(pairs_timestamp)):
        q_t, r_t, label_t = pair_t
        view, dis = cal_two_frame_dis_yaw(query_dataset_dir, ref_dataset_dir, int(q_t), int(r_t))
        pair = (f"Autumn_mini_val/{q_t}.jpg", f"Night_mini_val/{r_t}.jpg", label_t)
        logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
        if (view < 30 and dis < 25):
            dist_1.append(pair)
            # logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
            if (view < 10 and dis < 5):
                dist_2.append(pair)
                # logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
                if (view < 5 and dis < 0.5):
                    dist_3.append(pair)
                    if (view < 2 and dis < 0.25):
                        dist_4.append(pair)
        else:
            dist_null.append(pair)
    queue.put([dist_1, dist_2, dist_3, dist_4, dist_null])
    # return dist_1, dist_2, dist_3, dist_4, dist_null


def concate_list(list_2D):
    """
    Assume uniformed length 2D list: N*M
    In our case, (20,5)
    """
    logger.debug(f"list_2D: {list_2D}")
    # import pdb; pdb.set_trace()
    N = len(list_2D)
    M = len(list_2D[0])
    concate_lists = [list_2D[0][i] for i in range(M)]
    for i in range(1,N):
        for j in range(M):
            concate_lists[j].extend(list_2D[i][j])

    # concate_list = [concate_lists[j].extend(list_2D[i][j]) for i in range(1,N) for j in range(M)]
    # for i in range(N):
    #     for j in range(M):
    #         concate_lists[i].extend(list_2D[i][j])
    return concate_lists
            

def split_dataset(pairs, query_dataset, ref_dataset, output_dir, num_process=8):
    query_dataset_dir = f"dataset/robotcar/{query_dataset}"
    ref_dataset_dir = f"dataset/robotcar/{ref_dataset}"
    pairs_loader = parse_pairs(pairs)
    pairs_timestamp = [(q.strip('.jpg').split('/')[-1], r.strip('.jpg').split('/')[-1], label) for q, r, label in pairs_loader]
    # dist_1 = []
    # dist_2 = []
    # dist_3 = []
    # dist_4 = []
    # dist_null = []
    pairs_timestamp = [pairs_timestamp[i::num_process] for i in range(num_process)]
    q = Queue()
    processes = []
    dists = []
    # rets = [[dist_1, dist_2, dist_3, dist_4, dist_null]]
    for i in range(num_process):
            p = Process(target=split_view, args=(q, pairs_timestamp[i], query_dataset_dir, ref_dataset_dir))
            processes.append(p)
            p.start()
    for p in processes:
        dist = q.get()
        dists.append(dist)
    for p in processes:
        p.join()
        
    # import pdb; pdb.set_trace()
    # print(dist, len(dists))
    concate_lists = concate_list(dists)
    return concate_lists, len(concate_lists)
    
    
    # for pair_t in tqdm(pairs_timestamp, total = len(pairs_timestamp)):
        
    #     # q_t, r_t, label_t = pair_t
    #     # view, dis = cal_two_frame_dis_yaw(query_dataset_dir, ref_dataset_dir, int(q_t), int(r_t))
    #     # pair = (f"Autumn_mini_val/{q_t}.jpg", f"Suncloud_mini_val/{r_t}.jpg", label_t)
    #     # logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
    #     # if (view < 30 and dis < 25):
    #     #     dist_1.append(pair)
    #     #     # logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
    #     #     if (view < 10 and dis < 5):
    #     #         dist_2.append(pair)
    #     #         # logger.debug(f"view:{view}, dist:{dis}, pair:{pair}")
    #     #         if (view < 5 and dis < 0.5):
    #     #             dist_3.append(pair)
    #     #             if (view < 2 and dis < 0.25):
    #     #                 dist_4.append(pair)
    #     # else:
    #     #     dist_null.append(pair)
            
    # if dist_1 is not None:
    #     logger.info(f"dist_1: {len(dist_1)}")
    #     dist_1_dir = Path(output_dir, "qAutumn_db_Sumcloud_dist_1.txt")
    #     if not dist_1_dir.exists():
    #         dist_1_dir.parent.mkdir(parents=True)
    #     write_pairs(str(dist_1_dir), dist_1)
    # if dist_2 is not None:
    #     logger.info(f"dist_2: {len(dist_2)}")
    #     dist_2_dir = Path(output_dir, "qAutumn_db_Sumcloud_dist_2.txt")
    #     if not dist_2_dir.parent.exists():
    #         dist_2_dir.mkdir(parents=True)
    #     write_pairs(str(dist_2_dir), dist_2)
    # if dist_3 is not None:
    #     logger.info(f"dist_3: {len(dist_3)}")
    #     dist_3_dir = Path(output_dir, "qAutumn_db_Sumcloud_dist_3.txt")
    #     if not dist_3_dir.parent.exists():
    #         dist_3_dir.mkdir(parents=True)
    #     write_pairs(str(dist_3_dir), dist_3)
    # if dist_4 is not None:
    #     logger.info(f"dist_4: {len(dist_4)}")
    #     dist_4_dir = Path(output_dir, "qAutumn_db_Sumcloud_dist_4.txt")
    #     if not dist_4_dir.exists():
    #         dist_4_dir.parent.mkdir(parents=True)
    #     write_pairs(str(dist_4_dir), dist_4)
    # if dist_null is not None:
    #     logger.info(f"dist_null: {len(dist_null)}")
    #     dist_null_dir = Path(output_dir, "qAutumn_db_Sumcloud_dist_null.txt")
    #     if not dist_null_dir.parent.exists():
    #         dist_null_dir.mkdir(parents=True)
    #     write_pairs(str(dist_null_dir), dist_null)
    
    # return dist_1, dist_2, dist_3, dist_4, dist_null
    

def write_dists(dist_list, output_dir):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)
    N = 0
    for dist in dist_list:
        if dist is not None:
            logger.info(f"{len(dist)}")
            dist_dir = Path(output_dir, f"robotcar_qAutumn_dbNight_dist_{N}.txt")
            # if not dist_dir.exists():
            #     dist_dir.parent.mkdir(parents=True)
            write_pairs(str(dist_dir), dist)
        N += 1


# def write_gts(gts, output_dir):
#     if not Path(output_dir).exists():
#         Path(output_dir).mkdir(parents=True)
    
    
    

if __name__ == '__main__':
    from gvbench.utils import write_to_pairs
    gt = "dataset/robotcar/gt/robotcar_qAutumn_dbNight.txt"
    pairs = "dataset/robotcar/pairs/qAutumn_dbNight.txt"
    if not Path(pairs).parent.exists():
        Path(pairs).parent.mkdir(parents=True)
    write_to_pairs(gt, pairs)
    
    
    # gts, num_gts = gt_gen_multiprocess("dataset/robotcar/gt/robotcar_qAutumn_dbSuncloud.txt", "Autumn_val", "Suncloud_val", "dataset/robotcar/gt", 20)
    # # np.save("dataset/robotcar/gt/robotcar_qAutumn_dbSuncloud_dist.npy", np.array(dist_1))
    # print(f'# of gts: {num_gts}')
    # write_pairs("dataset/robotcar/gt/robotcar_qAutumn_dbSuncloud_new.txt", gts)
    # import pdb; pdb.set_trace()
    # dist_list = concate_list(dist_1)
    # np.save("dataset/robotcar/gt/robotcar_qAutumn_dbSuncloud_dist.npy", dist_list)
    # write_dists(dist_list, "dataset/robotcar/gt")
    
    # print(dist_1, dist_2, dist_3, dist_4)
    # cal_two_frame_dis_yaw("dataset/robotcar/Autumn_val", "dataset/robotcar/Night_val", 1418134268546931, 1418756758355180)

    # get mini dataset
    # matching_situation_fname = "robotcar_qAutumn_dbSunCloud_val_matchingSituation.txt"
    # matching_situation_mini_fname = "robotcar_qAutumn_dbSunCloud_val_mini.txt"
    # read_and_store(matching_situation_fname, matching_situation_mini_fname, "Autumn_val", "Autumn_mini_val", "Suncloud_val", "Suncloud_mini_val")

    # matching_situation_fname = "robotcar_qAutumn_dbNight_val_matchingSituation.txt"
    # matching_situation_mini_fname = "robotcar_qAutumn_dbNight_val_mini.txt"
    # read_and_store(matching_situation_fname, matching_situation_mini_fname, "Autumn_val", "Autumn_mini_val", "Night_val", "Night_mini_val")


    # construct dataset
    # matching_situation_mini_fnames = ["robotcar_qAutumn_dbSunCloud_val_mini.txt", "robotcar_qAutumn_dbNight_val_mini.txt"]
    # easy_fnames = ["robotcar_qAutumn_dbSunCloud_val_easy.txt", "robotcar_qAutumn_dbNight_val_easy.txt"]
    # diff_fnames = ["robotcar_qAutumn_dbSunCloud_val_diff.txt", "robotcar_qAutumn_dbNight_val_diff.txt"]
    # for matching_situation_mini_fname, easy_fname, diff_fname in zip(matching_situation_mini_fnames, easy_fnames, diff_fnames):
    #     construct_dataset(matching_situation_mini_fname, easy_fname, diff_fname)


    # get mini-mini dataset
    # easy_fname = "robotcar_qAutumn_dbSunCloud_val_easy.txt"
    # easy_final_fname = "robotcar_qAutumn_dbSunCloud_val_easy_final.txt"
    # store_mini_dataset(easy_fname, easy_final_fname, "Autumn_val", "Suncloud_val", "Autumn_mini_val_query", "Suncloud_mini_val_ref", True)

    # diff_fname = "robotcar_qAutumn_dbSunCloud_val_diff.txt"
    # diff_final_fname = "robotcar_qAutumn_dbSunCloud_val_diff_final.txt"
    # store_mini_dataset(diff_fname, diff_final_fname, "Autumn_val", "Suncloud_val", "Autumn_mini_val_query", "Suncloud_mini_val_ref", True)

    # easy_fname = "robotcar_qAutumn_dbNight_val_easy.txt"
    # easy_final_fname = "robotcar_qAutumn_dbNight_val_easy_final.txt"
    # store_mini_dataset(easy_fname, easy_final_fname, "Autumn_val", "Night_val", "Autumn_mini_val_query", "Night_mini_val_ref", False)

    # diff_fname = "robotcar_qAutumn_dbNight_val_diff.txt"
    # diff_final_fname = "robotcar_qAutumn_dbNight_val_diff_final.txt"
    # store_mini_dataset(diff_fname, diff_final_fname, "Autumn_val", "Night_val", "Autumn_mini_val_query", "Night_mini_val_ref", False)


    # loop_verification("IMG_3563.jpg", "IMG_3564.jpg", (640,480))
    # loop_verification("Autumn_mini/1418133688625998.jpg", "Suncloud_mini/1416318352045654.jpg", None, True)

    # glance_img("robotcar_qAutumn_dbSunCloud_val_easy_final.txt")
    # glance_img("robotcar_qAutumn_dbNight_val_diff_final.txt")

    # concatenate_txt("robotcar_qAutumn_dbSunCloud_val_easy_final.txt", "robotcar_qAutumn_dbSunCloud_val_diff_final.txt", "robotcar_qAutumn_dbSunCloud_val_final.txt")
    # concatenate_txt("robotcar_qAutumn_dbNight_val_easy_final.txt", "robotcar_qAutumn_dbNight_val_diff_final.txt", "robotcar_qAutumn_dbNight_val_final.txt")






    
