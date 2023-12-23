import os
import cv2
import json
import shutil
import random
import pickle
import imageio
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
from pandarallel import pandarallel
import matplotlib.pyplot as plt

data_dir = '/data/jupiter/li.yu/data/Jupiter_rev1_train_1784_human_sequences'
df = pd.read_csv(os.path.join(data_dir, 'master_annotations.csv'))
print(df.shape)

# pred_csv = '/home/li.yu/exps/driveable_terrain_model/v511rd_7cls_ft_dustaugonhuman_0908/Jupiter_rev1_train_1784_human_sequences/output.csv'
# pred_df = pd.read_csv(pred_csv)
# print(pred_df.shape)
# pred_df.groupby('human_state').count()

# gsam_results_dir = '/home/li.yu/data/Jupiter_train_v4_53_missing_human_relabeled/gsam_results_rectified/rectified_left/'
gsam_results_dir = f'{data_dir}/gsam_results_rectified/rectified_left/'
npzs = [f for f in os.listdir(gsam_results_dir) if f.endswith('.npz')]
print(len(npzs))

# parallel processing
def process(gsam_results_dir, row):
    human_threshold1, human_threshold2 = 120, 400
    row['id'] = row.npz.split('_')[0]
    try:
        data = np.load(os.path.join(gsam_results_dir, row.npz), allow_pickle=True)
        seg = data['gsam_mapped_8class_segmentation']
        human_count = np.count_nonzero(seg == 1)
        row['predicted_humans1'] = human_count >= human_threshold1
        row['predicted_humans2'] = human_count >= human_threshold2
    except:
        print(f"{row['npz']} file broken")
    return row

gsam_df = pd.DataFrame(data={'npz':npzs})
gsam_df['predicted_humans1'] = False
gsam_df['predicted_humans2'] = False
num_workers = 16
pandarallel.initialize(nb_workers=num_workers, progress_bar=False)
gsam_df = gsam_df.parallel_apply(lambda row: process(gsam_results_dir, row), axis=1)
print(gsam_df.shape)
gsam_df.to_csv(f'{data_dir}/gsam_results_rectified/rectified_left/summary.csv', index=False)
