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
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
from pandarallel import pandarallel
import matplotlib.pyplot as plt
sys.path.append('../')
from utils import normalize_image, plot_image, plot_images

# # root_dir = '/data/jupiter/li.yu/data'
# # root_dir = '/data/jupiter/datasets/'
# # root_dir = '/data2/jupiter/datasets/'
# root_dir = '/data3/jupiter/datasets/'
# # root_dir = '/data/jupiter/datasets/safety_datasets/'
# # dataset = 'halo_rgb_stereo_train_v10_0'
# dataset = 'halo_rgb_stereo_train_v11_3'
# # dataset = 'humans/on_path_aft/on_path_aft_humans_day_2024_rev2_v16'
# # csv = os.path.join(root_dir, dataset, 'annotations.csv')
# # csv = os.path.join(root_dir, dataset, 'master_annotations.csv')
# # csv = os.path.join(root_dir, dataset, 'master_annotations_more_drops_cleaned_20240611_rev1_lying_down_sitting_n_stop_events_n_night_dust_rev2_no_drops_w_label_counts.csv')
# csv = os.path.join(root_dir, dataset, 'master_annotations_full_cleaned_rev1_train_human_test_rev1_stops_22kdust_20240924.csv')
# # converters = {"label_map": ast.literal_eval, "label_counts": ast.literal_eval}
# converters = {"label_map": ast.literal_eval}
# # converters = {}
# df = pd.read_csv(csv, converters=converters)
# print(df.shape)

# # filter out ids already computed
# old_label_count_csv = '/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v11_1_categorical_count.csv'
# old_label_count_df = pd.read_csv(old_label_count_csv)
# print(old_label_count_df.shape)
# df = df[~df.unique_id.isin(old_label_count_df.unique_id)]
# print(df.shape)

# # get label counts
# df = df[['unique_id', 'label_map', 'rectified_label_save_path']]
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
#     label = np.load(os.path.join(root_dir, dataset, row.rectified_label_save_path))['left']
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
# # df.to_csv(os.path.join(root_dir, dataset, 'master_annotations_full_cleaned_rev1_train_human_test_rev1_stops_wlc.csv'), index=False)
# df = pd.concat([df[['unique_id'] + cats], old_label_count_df], ignore_index=True)
# df[['unique_id'] + cats].to_csv('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v11_3_categorical_count.csv', index=False)

# # get debayered rgb paths
# df = df[['unique_id', 'id', 'hdr_mode', 'artifact_debayeredrgb_0_save_path', 'stereo_left_image', 'label_save_path']]
# df['artifact_debayeredrgb_0_save_path'].fillna('', inplace=True)
# df['stereo_left_image'].fillna('', inplace=True)
# df['label_save_path'].fillna('', inplace=True)
# def check_raw_rgb_dir(root_dir, dataset, row):
#     row['raw_rgb_dir_exist'], row['raw_rgb_image_exist'], row['raw_rgb_label_exist'] = False, False, False
#     row['raw_rgb_image_size'], row['raw_rgb_label_size'] = [], []
#     try:
#         row['raw_rgb_dir_exist'] = os.path.isdir(os.path.join(root_dir, dataset, 'images', row.id))
#         if row.hdr_mode == True:
#             raw_rgb_image_path = os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path)
#         else:
#             raw_rgb_image_path = os.path.join(root_dir, dataset, row.stereo_left_image)
#         row['raw_rgb_image_exist'] = os.path.isfile(raw_rgb_image_path)
#         row['raw_rgb_label_exist'] = os.path.isfile(os.path.join(root_dir, dataset, row.label_save_path))
#         image = imageio.imread(raw_rgb_image_path)
#         row['raw_rgb_image_size'] = image.shape[:2]
#         label = imageio.imread(os.path.join(root_dir, dataset, row.label_save_path))
#         row['raw_rgb_label_size'] = label.shape
#     except:
#         pass
#     return row

# pandarallel.initialize(nb_workers=8, progress_bar=True)
# df = df.parallel_apply(lambda r: check_raw_rgb_dir(root_dir, dataset, r), axis=1)
# print(df.shape)
# df[['unique_id', 'raw_rgb_dir_exist', 'raw_rgb_image_exist', 'raw_rgb_label_exist', 'raw_rgb_image_size', 'raw_rgb_label_size']].to_csv('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v10_0_raw_rgb_exist_fixed.csv', index=False)


# # remove tiff files and save png to jpg format
# def convert_rgb_dataset(image_dir, jpg_dir, downsize, row):
#     img_dir = os.path.join(image_dir, row.id)
#     rgb_file = os.path.join(img_dir, f'artifact_debayeredrgb_0_{row.id}.png')
#     jpg_file = os.path.join(jpg_dir, f'{row.id}.jpg')
#     if os.path.isfile(jpg_file):
#         return
#     if os.path.isfile(rgb_file):
#         try:
#             img = cv2.imread(rgb_file)
#             if downsize:
#                 H, W, _ = img.shape
#                 cv2.imwrite(jpg_file, cv2.resize(img, (W//2,H//2)))
#             else:
#                 cv2.imwrite(jpg_file, img)
#         except:
#             pass
#     # if os.path.isdir(img_dir):
#     #     shutil.rmtree(img_dir)

# if len(sys.argv) > 1:
#     i_list = sys.argv[1:]
# for i in i_list:
#     # data_dir = f'/data3/jupiter/datasets/large_datasets/Jupiter_al_phase3_pool_pt{i}'
#     data_dir = f'/data3/jupiter/datasets/large_datasets/{i}'
#     image_dir = os.path.join(data_dir, 'images')
#     jpg_dir = os.path.join(data_dir, 'images_jpg')
#     os.makedirs(jpg_dir, exist_ok=True)

#     if len(os.listdir(image_dir)) == len(os.listdir(jpg_dir)):
#         continue

#     downsize = True if 'halo' in data_dir else False
#     files = os.listdir(image_dir)
#     df = pd.DataFrame(data={'id': files})
#     print(df.shape, f'downsize? {downsize}')
#     pandarallel.initialize(nb_workers=32, progress_bar=True)
#     df.parallel_apply(lambda r: convert_rgb_dataset(image_dir, jpg_dir, downsize, r), axis=1)


# check jpg files
def check_jpg_file(jpg_dir, row):
    row['corrupted'] = False
    try:
        Image.open(os.path.join(jpg_dir, row['img'])).convert("RGB")
    except:
        row['corrupted'] = True
    return row

if len(sys.argv) > 1:
    i_list = sys.argv[1:]
for i in range(int(i_list[0]), int(i_list[1]), 150):
    data_dir = f'/data2/jupiter/datasets/coyo-700m-webdataset/'
    jpg_dir = os.path.join(data_dir, f'tar_{i}_{int(i)+150}')
    # data_dir = f'/data3/jupiter/datasets/large_datasets/Jupiter_al_phase3_pool_pt{i}'
    # data_dir = f'/data3/jupiter/datasets/large_datasets/{i}'
    # jpg_dir = os.path.join(data_dir, 'images_jpg')

    files = [f for f in os.listdir(jpg_dir) if f.endswith('.jpg')]
    df = pd.DataFrame(data={'img': files})
    print(df.shape)
    pandarallel.initialize(nb_workers=32, progress_bar=True)
    df = df.parallel_apply(lambda r: check_jpg_file(jpg_dir, r), axis=1)
    df.to_csv(os.path.join(data_dir, f'tar_{i}_{int(i)+150}_corrupted.csv'), index=False)