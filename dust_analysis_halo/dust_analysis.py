import os
import ast
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
from datetime import datetime, timedelta, date
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import normalize_image, get_sequences

left_cameras = ['T01', 'T02', 'T05', 'T06', 'T09', 'T10', 'T13', 'T14', 'I01', 'I02']
camera_pairs = [['T01', 'T02'], ['T05', 'T06'], ['T09', 'T10'], ['T13', 'T14'], ['I01', 'I02']]

def read_raw_image(root_dir, dataset, row):
    return imageio.imread(os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path))

def read_rectified_image(root_dir, dataset, row):
    data_path = os.path.join(root_dir, dataset, row.stereo_pipeline_npz_save_path)
    img = np.load(data_path)['left']
    return (normalize_image(img, True)*255).astype(np.uint8)

def add_text(frame, raw_row, pred_df, dust_df):
    frame = cv2.putText(frame, f'Camera {raw_row.camera_location}, collected on: {raw_row.collected_on}', 
                        (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    pred_rows = pred_df[(pred_df.id == raw_row.id)]
    if len(pred_rows) > 1:
        s = 'Pred state in camera pair: '
        for i,r in pred_rows.iterrows():
            s += f'{r.unique_id[-7:]}: {r.state} '
        # print(raw_row.camera_location, pred_rows.unique_id.to_list())
        frame = cv2.putText(frame, s, 
                            (40,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    dust_rows = dust_df[dust_df.id == raw_row.id]
    if len(dust_rows) > 1:
        s = 'Pred dust ratio in camera pair: '
        for i,r in dust_rows.iterrows():
            s += f'{r.unique_id[-7:]}: {r.total_averaged_dust_conf} '
        # print(dust_rows.unique_id.to_list())
        frame = cv2.putText(frame, s, 
                            (40,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    return frame

def create_video_from_debayered_rgb(seq_dfs, si, raw_df, pred_root_dir, model, labeled_datasets, di, data_root_dir, unlabeled_datasets, pred_df, dust_df):
    seq_df = seq_dfs[si]

    # get raw sequence
    start, end = seq_df.iloc[0].collected_on, seq_df.iloc[-1].collected_on
    seq_raw_df = raw_df[(raw_df.collected_on >= start) & (raw_df.collected_on <= end)]
    print(seq_df.shape, seq_raw_df.shape)

    # get pair-wise cameras
    cameras = seq_df.camera_location.unique()
    cameras_full = set()
    for c in cameras:
        for cp in camera_pairs:
            if c in cp:
                cameras_full.add(cp[0])
                cameras_full.add(cp[1])
    cameras_full = list(cameras_full)
    cameras_full.sort()
    print(cameras, cameras_full)

    # get per-camera dfs and truncate to same length
    camera_dfs = [seq_raw_df[seq_raw_df.camera_location == c] for c in cameras_full]
    camera_dfs = [cdf.sort_values('collected_on', ignore_index=True) for cdf in camera_dfs]
    min_len = min(len(cdf) for cdf in camera_dfs)
    camera_dfs = [cdf.iloc[:min_len] for cdf in camera_dfs]
    print([len(cdf) for cdf in camera_dfs])

    # create video
    video_dir = os.path.join(pred_root_dir, model, labeled_datasets[di], 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_name = os.path.join(video_dir, f'{start}_{"_".join(cameras_full)}_{si}.mp4')
    frame = read_raw_image(data_root_dir, unlabeled_datasets[di], seq_df.iloc[5])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 3, (width*2,height*len(camera_dfs)//2), isColor=True)
    for fi in range(min_len):
        frames = []
        for pair_i in range(len(camera_dfs)//2):
            pair_frame = []
            for _fi in range(2):
                frame = read_raw_image(data_root_dir, unlabeled_datasets[di], camera_dfs[pair_i*2+_fi].iloc[fi])
                frame = add_text(frame, camera_dfs[pair_i*2+_fi].iloc[fi], pred_df, dust_df)
                pair_frame.append(frame)
            pair_frame = np.concatenate(pair_frame, axis=1)
            frames.append(pair_frame)
        frame = np.concatenate(frames, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    # cv2.destroyAllWindows()
    video.release()