import os
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
from datetime import datetime, timedelta, date
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from utils import normalize_image, get_sequences, plot_image, plot_images
from dust_analysis import left_pass_pairs, right_pass_pairs, read_raw_image, read_rectified_image, print_seg_performance, print_seg_dust_performance, get_seg_dust_performance_by_time


# root_dir = '/data/jupiter/li.yu/data'
# root_dir = '/data/jupiter/datasets/'
# root_dir = '/data2/jupiter/datasets/'
# root_dir = '/data2/jupiter/datasets/rev1_ask_ben_cline_before_deleting'
root_dir = '/data2/jupiter/datasets/oncal_fix/'
# dataset = 'halo_failure_case_of_box_in_dust'
# dataset = 'humans_on_path_test_set_2023_v15_anno'
# dataset = 'halo_vehicles_driving_through_dust_images_nodust_reserved_labeled'
dataset = 'halo_rgb_stereo_train_v8_1_max_fov_alleysson'
# csv = os.path.join(root_dir, dataset, 'master_annotations_mhc.csv')
csv = os.path.join(root_dir, dataset, 'annotations.csv')
# csv = os.path.join(root_dir, dataset, 'master_annotations.csv')
converters = {"label_map": ast.literal_eval, "label_counts": ast.literal_eval}
df = pd.read_csv(csv, converters=converters)
print(df.shape)


# # load in blurry / non-blurry binary labels
# blurry_label_file = '/data/jupiter/datasets/halo_failure_case_of_box_in_dust/halo_failure_case_of_box_in_dust_selected_for_binary_label.ndjson'
# bin_df = pd.read_json(blurry_label_file, lines=True)
# print(bin_df.shape)

# bin_df['id'] = bin_df.data_row.apply(lambda d: d['external_id'].split(',')[0])
# bin_df['iq_blurry'] = bin_df.projects.apply(lambda d: 'iq_blurry' in str(d))
# print(bin_df.groupby('iq_blurry').size())

# # assign back to original df, those missing binary labels are non-blurry automatically
# df['iq_blurry'] = False
# df.loc[df.id.isin(bin_df[bin_df.iq_blurry == True].id), "iq_blurry"] = True
# print(df.groupby('iq_blurry').size())


def read_raw_rgb(root_dir, dataset, row):
    return imageio.imread(os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path))

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def variance_of_laplacian_from_file(root_dir, dataset, row):
    try:
        image = cv2.imread(os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0

if not 'variance_of_laplacian' in df:
    # iq_blurry_csv = os.path.join(root_dir, dataset, 'iq_blurry.csv')
    iq_blurry_csv = os.path.join('/data/jupiter/li.yu/data/halo_rgb_stereo_train_test', 'train_v8_1_iq_blurry.csv')
    if os.path.isfile(iq_blurry_csv):
        iq_blurry_df = pd.read_csv(iq_blurry_csv)
        df = df.merge(iq_blurry_df, on='id')
    else:
        # df['variance_of_laplacian'] = df.apply(lambda r: variance_of_laplacian_from_file(root_dir, dataset, r), axis=1)
        pandarallel.initialize(nb_workers=8, progress_bar=False)
        df['variance_of_laplacian'] = df.parallel_apply(lambda r: variance_of_laplacian_from_file(root_dir, dataset, r), axis=1)

        cols = ['id', 'variance_of_laplacian']
        if 'iq_blurry' in df:
            cols.append('iq_blurry')
        df[cols].to_csv(iq_blurry_csv, index=False)