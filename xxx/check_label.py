import os
import sys
import ast
import cv2
import json
import math
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
sys.path.append('../')
from utils import normalize_image, plot_image, plot_images

# root_dir = '/data/jupiter/li.yu/data'
# root_dir = '/data/jupiter/datasets/'
root_dir = '/data2/jupiter/datasets/'
# root_dir = '/data/jupiter/datasets/safety_datasets/'
dataset = 'halo_rgb_stereo_train_v10_0'
# dataset = 'humans/on_path_aft/on_path_aft_humans_day_2024_rev2_v16'
# csv = os.path.join(root_dir, dataset, 'annotations.csv')
# csv = os.path.join(root_dir, dataset, 'master_annotations.csv')
# csv = os.path.join(root_dir, dataset, 'master_annotations_more_drops_cleaned_20240611_rev1_lying_down_sitting_n_stop_events_n_night_dust_rev2_no_drops_w_label_counts.csv')
csv = os.path.join(root_dir, dataset, 'master_annotations_more_drops_cleaned_20240611_rev1_lying_down_sitting_n_stop_events_rev2_no_drops_w_label_counts_for_debayeredrgb.csv')
# converters = {"label_map": ast.literal_eval, "label_counts": ast.literal_eval}
# converters = {"label_map": ast.literal_eval}
converters = {}
df = pd.read_csv(csv, converters=converters)
print(df.shape)

# # get label counts
# categorical_labels_map = {'objects_pixel_count': {'Utility pole', 'Immovable Objects', 'Buildings', 'Animals', 'Tile-Inlet'}, 
#                           'humans_pixel_count': {'Humans'}, 'tractors_or_vehicles_pixel_count': {'Tractors or Vehicles'}, 
#                           'dust_pixel_count': {'Heavy Dust'}, 'birds_pixel_count': {'Birds'}, 'airborne_debris_pixel_count': {'Airborne Debris'},
#                           'unharvested_field_pixel_count': {'Unharvested Field'}, 'trees_pixel_count': {'Trees'}}
# cats = list(categorical_labels_map.keys())
# categorical_object_labels = {v for k,vs in categorical_labels_map.items() for v in vs}
# print(cats, categorical_object_labels)

# def get_categorical_labels(root_dir, dataset, row):
#     # # raw label
#     # label = imageio.imread(os.path.join(root_dir, dataset, row.annotation_pixelwise_0_save_path))
#     # rectified label
#     label_path = os.path.join(root_dir, dataset, row.rectified_label_save_path)
#     label = np.load(label_path)['left']
#     labels = np.unique(label)
#     label_str_2_id = {row.label_map[str(i)]: i for i in labels if i != 0}
#     # print(row.unique_id, label.shape, labels, label_str_2_id)
#     # process object class
#     for object_label, subs in categorical_labels_map.items():
#         object_ids = [label_str_2_id[sub] for sub in subs if sub in label_str_2_id]
#         object_pixel_count = 0
#         if len(object_ids) > 0:
#             object_pixel_count = np.count_nonzero(np.isin(label, object_ids))
#         row[object_label] = object_pixel_count
#     return row

# pandarallel.initialize(nb_workers=8, progress_bar=True)
# df = df.parallel_apply(lambda r: get_categorical_labels(root_dir, dataset, r), axis=1)
# print(df.shape)
# df[['unique_id'] + cats].to_csv('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v10_0_categorical_count.csv', index=False)

df = df[['unique_id', 'id', 'hdr_mode', 'artifact_debayeredrgb_0_save_path', 'stereo_left_image', 'label_save_path']]
df['artifact_debayeredrgb_0_save_path'].fillna('', inplace=True)
df['stereo_left_image'].fillna('', inplace=True)
df['label_save_path'].fillna('', inplace=True)
def check_raw_rgb_dir(root_dir, dataset, row):
    row['raw_rgb_dir_exist'], row['raw_rgb_image_exist'], row['raw_rgb_label_exist'] = False, False, False
    row['raw_rgb_image_size'], row['raw_rgb_label_size'] = [], []
    try:
        row['raw_rgb_dir_exist'] = os.path.isdir(os.path.join(root_dir, dataset, 'images', row.id))
        if row.hdr_mode == True:
            raw_rgb_image_path = os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path)
        else:
            raw_rgb_image_path = os.path.join(root_dir, dataset, row.stereo_left_image)
        row['raw_rgb_image_exist'] = os.path.isfile(raw_rgb_image_path)
        row['raw_rgb_label_exist'] = os.path.isfile(os.path.join(root_dir, dataset, row.label_save_path))
        image = imageio.imread(raw_rgb_image_path)
        row['raw_rgb_image_size'] = image.shape[:2]
        label = imageio.imread(os.path.join(root_dir, dataset, row.label_save_path))
        row['raw_rgb_label_size'] = label.shape
    except:
        pass
    return row

pandarallel.initialize(nb_workers=8, progress_bar=True)
df = df.parallel_apply(lambda r: check_raw_rgb_dir(root_dir, dataset, r), axis=1)
print(df.shape)
df[['unique_id', 'raw_rgb_dir_exist', 'raw_rgb_image_exist', 'raw_rgb_label_exist', 'raw_rgb_image_size', 'raw_rgb_label_size']].to_csv('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v10_0_raw_rgb_exist_fixed.csv', index=False)
