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
dataset = 'rev2_rgb_data_9_2_train_20240607_stereo'
# dataset = 'humans/on_path_aft/on_path_aft_humans_day_2024_rev2_v16'
# csv = os.path.join(root_dir, dataset, 'annotations.csv')
# csv = os.path.join(root_dir, dataset, 'master_annotations.csv')
csv = os.path.join(root_dir, dataset, 'master_annotations_more_drops_cleaned_20240611_rev1_lying_down_sitting_drop_dups.csv')
# converters = {"label_map": ast.literal_eval, "label_counts": ast.literal_eval}
converters = {"label_map": ast.literal_eval}
# converters = {}
df = pd.read_csv(csv, converters=converters)
print(df.shape)

categorical_labels_map = {'objects': {'Utility pole', 'Immovable Objects', 'Buildings', 'Animals', 'Tile-Inlet'}, 'humans': {'Humans'}, 
                            'vehicles': {'Tractors or Vehicles'}, 'dust': {'Heavy Dust'}, 'birds': {'Birds'}, 'airborne': {'Airborne Debris'},
                            'unharvested_field': {'Unharvested Field'}, 'trees': {'Trees'}}
cats = list(categorical_labels_map.keys())
categorical_object_labels = {v for k,vs in categorical_labels_map.items() for v in vs}
cats, categorical_object_labels

def get_categorical_labels(root_dir, dataset, row):
    # # raw label
    # label = imageio.imread(os.path.join(root_dir, dataset, row.annotation_pixelwise_0_save_path))
    # rectified label
    label_path = os.path.join(root_dir, dataset, row.rectified_label_save_path)
    label = np.load(label_path)['left']
    labels = np.unique(label)
    label_str_2_id = {row.label_map[str(i)]: i for i in labels if i != 0}
    # print(row.unique_id, label.shape, labels, label_str_2_id)
    # process object class
    for object_label, subs in categorical_labels_map.items():
        object_ids = [label_str_2_id[sub] for sub in subs if sub in label_str_2_id]
        object_pixel_count = 0
        if len(object_ids) > 0:
            object_pixel_count = np.count_nonzero(np.isin(label, object_ids))
        row[object_label] = object_pixel_count
    return row

pandarallel.initialize(nb_workers=8, progress_bar=True)
df = df.parallel_apply(lambda r: get_categorical_labels(root_dir, dataset, r), axis=1)
print(df.shape)
df[['unique_id'] + cats].to_csv('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v9_2_categorical_count.csv', index=False)
