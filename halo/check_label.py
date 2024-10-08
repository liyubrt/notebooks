import os
import sys
import cv2
import json
import ndjson
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

sys.path.append('/home/li.yu/code/JupiterCVML/europa/base/src/europa')
from dl.utils.config import LEFT, MAX_DEPTH, POINT_CLOUD
from dl.utils.image_transforms import depth_from_point_cloud


# root_dir = '/data/jupiter/datasets'
root_dir = '/data2/jupiter/datasets'
# dataset = 'Jupiter_train_v6_2'
# dataset = 'humans_on_path_test_set_2023_v15_anno'
# dataset = 'halo_rgb_stereo_train_v6_1'
# dataset = 'halo_rgb_stereo_test_v6_1'
# dataset = 'halo_rgb_stereo_train_v6_2_full_res'
# dataset = 'halo_rgb_stereo_test_v6_2'
# dataset = 'halo_humans_on_path_test_v6_2_3_mainline'
# dataset = 'halo_rgb_stereo_train_v8_0'
# dataset = 'halo_rgb_stereo_test_v8_0'
dataset = 'halo_rgb_stereo_train_v10_0'
csv = os.path.join(root_dir, dataset, 'master_annotations.csv')
df = pd.read_csv(csv)
print(df.shape)

# compute relevant columns
columns=[
    "image_size", 
    "is_human_present", "human_pixels", "human_min_row", "human_max_row", "human_min_col", "human_max_col", "human_median_depth", "human_90_percentile_depth", "human_max_depth", 
    "is_vehicle_present", "vehicle_pixels", "vehicle_min_row", "vehicle_max_row", "vehicle_min_col", "vehicle_max_col", "vehicle_median_depth", "vehicle_90_percentile_depth", "vehicle_max_depth", 
]


def get_object_location(object_mask: np.ndarray):
    rows, cols = np.where(object_mask)
    return np.min(rows), np.max(rows), np.min(cols), np.max(cols)

def process_object(label_mask, depth, prefix='human'):
    res = {}
    res[f'{prefix}_pixels'] = np.count_nonzero(label_mask)
    res[f'{prefix}_min_row'], res[f'{prefix}_max_row'], res[f'{prefix}_min_col'], res[f'{prefix}_max_col'] = get_object_location(label_mask)
    depth_values = depth[label_mask]
    res[f'{prefix}_median_depth'] = np.median(depth_values)
    res[f'{prefix}_90_percentile_depth'] = np.percentile(depth_values, q=90)
    res[f'{prefix}_max_depth'] = np.max(depth_values)
    res[f'is_{prefix}_present'] = 'Yes'
    return res

def get_human_vehicle_info(root_dir, dataset, row):
    res = {c:'' for c in columns}
    # get depth
    stereo_data_sample = np.load(os.path.join(root_dir, dataset, row.stereo_pipeline_npz_save_path))
    depth = depth_from_point_cloud(
        stereo_data_sample[POINT_CLOUD],
        clip_and_normalize=True,
        max_depth=MAX_DEPTH,
        make_3d=True,
    )[:,:,0]
    # get label
    label_path = os.path.join(root_dir, dataset, row.rectified_label_save_path)
    label = np.load(label_path)['left'][:,:,0]
    labels = np.unique(label)
    label_map = json.loads(row.label_map)
    # get objects
    res['image_size'] = label.shape
    for i in labels:
        if i == 0:
            continue
        if label_map[str(i)] == "Humans":
            res.update(process_object(label == i, depth, 'human'))
        if label_map[str(i)] == "Tractors or Vehicles":
            res.update(process_object(label == i, depth, 'vehicle'))
    # add to row
    for k,v in res.items():
        row[k] = v
    return row

pandarallel.initialize(nb_workers=8, progress_bar=True)
df = df.parallel_apply(lambda r: get_human_vehicle_info(root_dir, dataset, r), axis=1)
print(df.shape)
df[columns].to_csv('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v10_0.csv', index=False)