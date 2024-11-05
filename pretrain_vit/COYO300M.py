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
import logging
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
from pandarallel import pandarallel
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
sys.path.append('../')
from utils import normalize_image, plot_image, plot_images
logging.basicConfig(level=logging.INFO)


def find_matches(pi, data_dir, label_parquets):
    df = pd.read_parquet(f'{data_dir}/huggingface/data/{label_parquets[pi]}')
    df = df[['url', 'imagehash', 'labels', 'label_probs']]
    for part1m_i in range(145):
        M, N = part1m_i*150, (part1m_i+1)*150
        match_csv = os.path.join(data_dir, 'matches', f'{pi}_{M}_{N}.csv')
        # part1m_dir = os.path.join(data_dir, f'tar_{M}_{N}')
        if os.path.isfile(match_csv):
            continue
        # dfs2 = [pd.read_parquet(f'{data_dir}/metadata/{str(part10k_i).zfill(5)}.parquet') for part10k_i in range(M, N)]
        dfs2 = []
        for part10k_i in range(M, N):
            park10k_i_csv = f'{data_dir}/metadata/{str(part10k_i).zfill(5)}.parquet'
            try:
                _df = pd.read_parquet(park10k_i_csv)
                dfs2.append(_df[['url', 'caption', 'key']])
            except:
                print(f'corrupted file: {park10k_i_csv}')
        df2 = pd.concat(dfs2, ignore_index=True)
        df3 = df.merge(df2, on='url')
        if len(df3) > 0:
            df3['part1m_dir'] = f'tar_{M}_{N}'
            df3.to_csv(match_csv, index=False)
        print(f'label {pi} partition: checked folder tar_{M}_{N} and found {len(df3)} matches')

def isfile(row, data_dir):
    row['downloaded'] = os.path.isfile(os.path.join(data_dir, row.part1m_dir, f'{str(row.key).zfill(9)}.jpg'))
    return row

def check_download(part1m_i, data_dir):
    M, N = part1m_i*150, (part1m_i+1)*150
    label_csv = os.path.join(data_dir, 'matches_downloaded', f'label_{M}_{N}.csv')
    dfs = []
    for pi in range(128):
        match_csv = os.path.join(data_dir, 'matches', f'{pi}_{M}_{N}.csv')
        if not os.path.isfile(match_csv):
            continue
        df = pd.read_csv(match_csv)
        df['downloaded'] = False
        df = df.apply(lambda r: isfile(r, data_dir), axis=1)
        df = df[df.downloaded == True]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f'downloaded {len(df)} images with labels for tar_{M}_{N}') 
    df.to_csv(label_csv, index=False)

# check jpg files
import warnings
warnings.filterwarnings('error')  # catch warning as an error
def check_jpg_file(data_dir, row):
    try:
        Image.open(os.path.join(data_dir, row.part1m_dir, f'{str(row.key).zfill(9)}.jpg')).convert('RGB')
    except:
        row['corrupted'] = True
    if row.key % 10000 == 0:
        logging.info(f'processing image key {row.key}')
    return row

data_dir = '/data2/jupiter/datasets/coyo-700m-webdataset'

# executor = ProcessPoolExecutor(max_workers=16)
# tasks = []

# # label_parquets = os.listdir(f'{data_dir}/huggingface/data')
# # print(len(label_parquets))
# # for pi in range(128):
# #     tasks.append(executor.submit(find_matches, pi, data_dir, label_parquets))
# #     # find_matches(pi, data_dir, label_parquets)

# for part1m_i in range(145):
#     tasks.append(executor.submit(check_download, part1m_i, data_dir))
#     # check_download(part1m_i, data_dir)

# job_count = len(tasks)
# for future in as_completed(tasks):
#     job_count -= 1
#     print(f'Remaining jobs: {job_count}')

all_label_df = pd.read_parquet(os.path.join(data_dir, 'matches_downloaded', 'label_02p.parquet'))

# with open(os.path.join(data_dir, 'matches_downloaded', 'label_counts.pkl'), 'rb') as f:
#     big_counts = pickle.load(f)
#     big_count_ids = pickle.load(f)
#     small_counts = pickle.load(f)
#     small_count_ids = pickle.load(f)
# print(len(big_counts), len(big_count_ids))
# print(len(small_counts), len(small_count_ids))

# # get one hot encoding and check class distribution relative to big counts
# def check_label(row, big_count_ids):
#     one_hot_count = 0
#     for i in row.labels_filtered:
#         if i in big_count_ids:
#             one_hot_count += 1
#     row['one_hot_small_count'] = one_hot_count
#     if row.key % 10000 == 0:
#         print(f'processing image key {row.key}')
#     return row

# pandarallel.initialize(nb_workers=16, progress_bar=False)
# all_label_df['one_hot_small_count'] = 0
# all_label_df = all_label_df.parallel_apply(lambda r: check_label(r, small_count_ids), axis=1)
# # all_label_df = all_label_df.apply(lambda r: check_label(r, big_count_ids), axis=1)

pandarallel.initialize(nb_workers=16, progress_bar=False)
all_label_df['corrupted'] = False
all_label_df = all_label_df.parallel_apply(lambda r: check_jpg_file(data_dir, r), axis=1)

all_label_df.to_parquet(os.path.join(data_dir, 'matches_downloaded', 'label_02p.parquet'), index=False)