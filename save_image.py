import os
import cv2
import json
import shutil
import random
import pickle
import imageio
import logging
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import sys
sys.path.append('./')
from utils import normalize_image, get_sequences


# data_root_dir = '/data/jupiter/li.yu/data'
# unlabeled_datasets = ["Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2", 
#                       "Jupiter_2023_04_05_loamy869_dust_collection_stereo", 
#                       "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo"]
# labeled_datasets = ["Jupiter_2023_03_02_and_2930_human_vehicle_in_dust_labeled", 
#                     "Jupiter_2023_March_29th30th_human_vehicle_in_dust_front_pod_labeled", 
#                     "Jupiter_2023_04_05_loamy869_dust_collection_stereo_labeled", 
#                     "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo_labeled"]
# pred_root = '/data/jupiter/li.yu/exps/driveable_terrain_model/'

# i = 0

# # os.makedirs(os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_images'), exist_ok=True)
# # master_df = pd.read_csv(os.path.join(data_root_dir, unlabeled_datasets[i], 'master_annotations.csv'), low_memory=False)
# # print(master_df.shape)
# # for idx, row in master_df.iterrows():
# #     data_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'processed/images', row.id, 'stereo_output.npz')
# #     img = np.load(data_path)['left']
# #     img_norm = normalize_image(img, row.hdr_mode)
# #     img_norm = (img_norm * 255).astype(np.uint8)
# #     img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
# #     save_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_images', row.id+'.png')
# #     cv2.imwrite(save_path, img_norm)
# #     if idx % 1000 == 0:
# #         print(f'saved {idx+1} images')

# os.makedirs(os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_labels'), exist_ok=True)
# master_df = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i], 'master_annotations.csv'), low_memory=False)
# print(master_df.shape)
# for idx, row in master_df.iterrows():
#     label_path = os.path.join(data_root_dir, labeled_datasets[i], row.rectified_label_save_path)
#     lbl = np.load(label_path)['left']
#     save_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_labels', row.id+'.png')
#     cv2.imwrite(save_path, lbl)
#     if idx % 1000 == 0:
#         print(f'saved {idx+1} images')

# master_df = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i+1], 'master_annotations.csv'), low_memory=False)
# print(master_df.shape)
# for idx, row in master_df.iterrows():
#     label_path = os.path.join(data_root_dir, labeled_datasets[i+1], row.rectified_label_save_path)
#     lbl = np.load(label_path)['left']
#     save_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_labels', row.id+'.png')
#     cv2.imwrite(save_path, lbl)
#     if idx % 1000 == 0:
#         print(f'saved {idx+1} images')



# # root_dir = '/data/jupiter/li.yu/data'
# root_dir = '/data/jupiter/datasets/dust_datasets'
# # root_dir = '/data2/jupiter/datasets/'
# dataset = 'halo_dust_on_lens_blur_dataset_v3_20240807'
# # dataset = 'halo_rgb_stereo_train_v8_0'
# csv = os.path.join(root_dir, dataset, 'annotations.csv')
# # csv = os.path.join(root_dir, dataset, 'master_annotations.csv')
# # converters = {"label_map": ast.literal_eval, "label_counts": ast.literal_eval}
# converters = {}
# df = pd.read_csv(csv, converters=converters)
# logging.info(f'{df.shape}')

# sample_csv = os.path.join(root_dir, dataset, 'iq_fn_depth_smudge_halo_dust_on_lens_blur_dataset_v3_20240807.csv')
# sdf = pd.read_csv(sample_csv)
# sdf['id'] = sdf['unique_id'].apply(lambda s: s[:-8])
# df = df[df.id.isin(sdf.id)]
# logging.info(f'{df.shape}')

# save_dir = '/data/jupiter/datasets/20240301_5_million_for_self_supervised_part_0'
# rectified_dir = os.path.join(save_dir, 'rev1_train')
# os.makedirs(rectified_dir, exist_ok=True)
# for i,row in df.iterrows():
#     data_path = os.path.join(root_dir, dataset, row.stereo_pipeline_npz_save_path)
#     img = np.load(data_path)['left']
#     img_norm = normalize_image(img, hdr_mode=row.hdr_mode, return_8_bit=True)
#     Image.fromarray(img_norm).save(os.path.join(rectified_dir, row.unique_id+'.jpg'))
#     if (i+1) % 2000 == 0:
#         logging.info(f'processed {i+1} images')
# saved_ids = df.unique_id.to_list()
# saved_df = pd.DataFrame(data={'id': [f'rev1_train/{f}.jpg' for f in saved_ids]})
# saved_df.to_csv(os.path.join(save_dir, 'rev1_train_saved_ids.csv'), index=False)

# seq_dfs = get_sequences(df, interval=60)  # break the data by intervals between sequences
# print(df.shape, len(seq_dfs))
# all_cameras = {'front': ['T01', 'T02', 'T03', 'T04'], 'right': ['T05', 'T06', 'T07', 'T08'], 'back': ['T09', 'T10', 'T11', 'T12'], 'left': ['T13', 'T14', 'T15', 'T16']}
# save_dir = os.path.join(root_dir, dataset, 'all_in_seqs')
# os.makedirs(save_dir, exist_ok=True)
# for pod, cameras in all_cameras.items():
#     print(pod, cameras)
#     for i,seq_df in enumerate(seq_dfs):
#         cam_df = seq_df[seq_df.camera_location.isin(cameras)]
#         if len(cam_df) == 0:
#             continue
#         print(pod, i, len(cam_df))
#         sub_save_dir = os.path.join(save_dir, f'{pod}_{str(i).zfill(2)}')
#         os.makedirs(sub_save_dir, exist_ok=True)
#         for _, row in cam_df.iterrows():
#             img_path = os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path)
#             img = cv2.imread(img_path)
#             cv2.imwrite(os.path.join(sub_save_dir, f'{row.id}.jpg'), img)
#             # rename image
#             os.rename(os.path.join(sub_save_dir, f'{row.id}.jpg'), os.path.join(sub_save_dir, f'{row.camera_location}_{row.id}.jpg'))



# download images with urls
import requests
from pandarallel import pandarallel
def download_image(save_dir, row):
    file_path = os.path.join(save_dir, row['Subset'], row['ImageID']+'.jpg')
    if not os.path.isfile(file_path):
        try:
            img_data = requests.get(row['OriginalURL'], timeout=3).content
            with open(file_path, 'wb') as handler:
                handler.write(img_data)
        except:
            pass

save_dir = '/data3/jupiter/datasets/public_datasets/OpenImages'
for i in range(6, 10):
    csv_file = f'/data3/jupiter/datasets/public_datasets/OpenImages/meta_data/image_ids_and_rotation_part{i}.csv'
    df = pd.read_csv(csv_file, low_memory=False)
    print(df.shape)
    pandarallel.initialize(nb_workers=32, progress_bar=False)
    df.parallel_apply(lambda r: download_image(save_dir, r), axis=1)