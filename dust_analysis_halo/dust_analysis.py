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
from datetime import datetime, timedelta, date
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import normalize_image, get_sequences

left_cameras = ['T01', 'T02', 'T05', 'T06', 'T09', 'T10', 'T13', 'T14', 'I01', 'I02']
camera_pairs = [['T01', 'T02'], ['T05', 'T06'], ['T09', 'T10'], ['T13', 'T14'], ['I01', 'I02']]
left_pass_pairs = ['T09_T11', 'T14_T16', 'T14_T15', 'T13_T15']
right_pass_pairs = ['T05_T07', 'T10_T12', 'T06_T08', 'T06_T07']

def print_seg_performance(pred_df, model_desc):
    tp = sum(pred_df.state == 'true_positive')
    tn = sum(pred_df.state == 'true_negative')
    fp = sum(pred_df.state == 'false_positive')
    fn = sum(pred_df.state == 'false_negative')
    recall = tp / (tp + fn)
    productivity = tn / (tn + fp)
    print(f'{model_desc}: TP {tp}, TN {tn}, FP {fp}, FN {fn}, Recall {recall:.4f}, Productivity {productivity:.4f}')

def print_seg_dust_performance(pred_df, dust_df, model_desc):
    tp_df = pred_df[(pred_df.state == 'true_positive')]
    tn_df = pred_df[(pred_df.state == 'true_negative')]
    fp_df = pred_df[(pred_df.state == 'false_positive')]
    fn_df = pred_df[(pred_df.state == 'false_negative')]
    tp_dust_df = dust_df[dust_df.unique_id.isin(tp_df.unique_id)]
    tn_dust_df = dust_df[dust_df.unique_id.isin(tn_df.unique_id)]
    fp_dust_df = dust_df[dust_df.unique_id.isin(fp_df.unique_id)]
    fn_dust_df = dust_df[dust_df.unique_id.isin(fn_df.unique_id)]
    recall = len(tp_df) / (len(tp_df) + len(fn_df))
    productivity = len(tn_df) / (len(tn_df) + len(fp_df))
    print(f'{model_desc}: TP {len(tp_df)}, TN {len(tn_df)}, FP {len(fp_df)}, FN {len(fn_df)}, Recall {recall:.4f}, Productivity {productivity:.4f}')
    print(f'{model_desc} dust: TP {tp_dust_df.total_averaged_dust_conf.mean():.2f}, TN {tn_dust_df.total_averaged_dust_conf.mean():.2f}, FP {fp_dust_df.total_averaged_dust_conf.mean():.2f}, FN {fn_dust_df.total_averaged_dust_conf.mean():.2f}')

def get_seg_dust_performance_by_time(raw_df, pred_df, dust_df, model_desc):
    raw_df2 = raw_df.drop('state', axis=1)
    pred_df = pred_df.merge(raw_df2, on='unique_id')
    day_pred_df = pred_df[~pred_df.collected_on.str.startswith('2024-03-01')]
    night_pred_df = pred_df[pred_df.collected_on.str.startswith('2024-03-01')]
    print(day_pred_df.shape, night_pred_df.shape)
    print_seg_dust_performance(day_pred_df, dust_df, model_desc + ' day')
    print_seg_dust_performance(night_pred_df, dust_df, model_desc + ' night')

def read_raw_image(root_dir, dataset, row):
    return imageio.imread(os.path.join(root_dir, dataset, row.artifact_debayeredrgb_0_save_path))

def read_rectified_image(root_dir, dataset, row):
    data_path = os.path.join(root_dir, dataset, row.stereo_pipeline_npz_save_path)
    img = np.load(data_path)['left']
    return (normalize_image(img, True)*255).astype(np.uint8)

def add_text(frame, raw_row, pred_df, dust_df):
    frame = cv2.putText(frame, f'{raw_row.camera_pair}, {raw_row.collected_on}', 
                        (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    pred_rows = pred_df[pred_df.unique_id == raw_row.unique_id]
    if len(pred_rows) > 0:
        s = f'Pred state: {pred_rows.iloc[0].state}'
        frame = cv2.putText(frame, s, 
                            (40,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    dust_rows = dust_df[dust_df.unique_id == raw_row.unique_id]
    if len(dust_rows) > 0:
        s = f'Pred dust ratio: {dust_rows.iloc[0].total_averaged_dust_conf}'
        frame = cv2.putText(frame, s, 
                            (40,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    return frame

def create_video_from_rectified_rgb(seq_dfs, si, raw_df, pred_root_dir, model, labeled_datasets, di, data_root_dir, unlabeled_datasets, pred_df, dust_df):
    seq_df = seq_dfs[si]

    # get raw sequence
    start, end = seq_df.iloc[0].collected_on, seq_df.iloc[-1].collected_on
    seq_raw_df = raw_df[(raw_df.collected_on >= start) & (raw_df.collected_on <= end)]
    print(f'process {si+1}th sequence', seq_df.shape, seq_raw_df.shape)

    # get pair-wise cameras
    cameras = list(seq_df.camera_location.unique())
    cameras.sort()
    camera_pairs = list(seq_df.camera_pair.unique())
    camera_pairs.sort()
    print(cameras, camera_pairs)

    # check if should use left pass or right pass or both
    in_left = set(camera_pairs).intersection(left_pass_pairs)
    in_right = set(camera_pairs).intersection(right_pass_pairs)
    print(len(in_left), len(in_right))
    if len(in_left) > len(in_right):
        camera_pairs = left_pass_pairs
        pass_key = 'left_pass'
    elif len(in_left) < len(in_right):
        camera_pairs = right_pass_pairs
        pass_key = 'right_pass'
    else:
        pass_key = 'short_pass'

    # get per-camera dfs and truncate to same length
    camera_dfs = [seq_raw_df[seq_raw_df.unique_id.str.endswith(c)] for c in camera_pairs]
    camera_dfs = [cdf.sort_values('collected_on', ignore_index=True) for cdf in camera_dfs]
    min_len = min(len(cdf) for cdf in camera_dfs)
    camera_dfs = [cdf.iloc[:min_len] for cdf in camera_dfs]
    print([len(cdf) for cdf in camera_dfs])

    # create video
    video_dir = os.path.join(pred_root_dir, model, labeled_datasets[di], 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_name = os.path.join(video_dir, f'{start}_{pass_key}_{si}.mp4')
    height, width = 512, 768
    # print(width*2,height*math.ceil(len(camera_dfs)/2))

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 3, (width*2,height*math.ceil(len(camera_dfs)/2)), isColor=True)
    for fi in range(min_len):
        frames = []
        for pair_i in range(math.ceil(len(camera_dfs)/2)):
            pair_frame = []
            for _fi in range(2):
                if pair_i*2+_fi == len(camera_dfs):
                    frame = np.zeros((512, 768, 3), dtype=np.uint8)
                else:
                    frame = read_rectified_image(data_root_dir, unlabeled_datasets[di], camera_dfs[pair_i*2+_fi].iloc[fi])
                    if frame.shape[1] == 640:
                        zeros = np.zeros((512, 768, 3), dtype=frame.dtype)
                        zeros[:,:640,:] = frame
                        frame = zeros
                    frame = add_text(frame, camera_dfs[pair_i*2+_fi].iloc[fi], pred_df, dust_df)
                pair_frame.append(frame)
            pair_frame = np.concatenate(pair_frame, axis=1)
            frames.append(pair_frame)
        frame = np.concatenate(frames, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    # cv2.destroyAllWindows()
    video.release()