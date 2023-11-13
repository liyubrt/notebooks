#!/usr/bin/env python
# coding: utf-8

import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/li.yu/code/JupiterCVML/europa/base/src/europa')
sys.path = list(set(sys.path))

data_directory = '/data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST'
save_directory = '/data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST'
# left_tire_mask = '/home/li.yu/code/JupiterEmbedded/src/dnn_engine/data/side_left_mask.png'
# right_tire_mask = '/home/li.yu/code/JupiterEmbedded/src/dnn_engine/data/side_right_mask.png'
# left_tire_mask = '/data/jupiter/avinash.raju/tire_masks/side_left_new.png'
# right_tire_mask = '/data/jupiter/avinash.raju/tire_masks/side_right_new.png'

default_loc = '/data/jupiter/avinash.raju/tire_masks'
side_left_left_tire_mask = f'{default_loc}/side_left_new.png'
side_left_right_tire_mask = f'{default_loc}/side_left_new.png'
side_right_left_tire_mask = f'{default_loc}/side_right_new.png'
side_right_right_tire_mask = f'{default_loc}/side_right_new.png'

from cv.core.image_quality_server_side import ImageQuality

workers = mp.cpu_count() // 2
# iq = ImageQuality(num_workers=workers,
#                   use_progress=False,
#                   side_left_tire_mask_path = left_tire_mask,
#                   side_right_tire_mask_path = right_tire_mask,
#                   normalization_policy='percentile')
iq = ImageQuality(num_workers=workers,
                  use_progress=False,
                  side_left_left_tire_mask_path = side_left_left_tire_mask,
                  side_left_right_tire_mask_path = side_left_right_tire_mask,
                  side_right_left_tire_mask_path = side_right_left_tire_mask,
                  side_right_right_tire_mask_path = side_right_right_tire_mask,
                  normalization_policy='global_tonemap')
stereo_df = pd.read_csv(os.path.join(data_directory, 'master_annotations_labeled.csv'), low_memory=False)
# stereo_df = stereo_df.sample(1024)
print(stereo_df.shape)

print('start calculation')
labeled = iq.from_df(stereo_df, data_directory, use_progress=False)
print('finish calculation')

print('start column mapping')
print('process iq'); time.sleep(3);
labeled['iq'] = labeled.image_quality.parallel_apply(lambda x: x.algorithm_output)
print('process iq_features'); time.sleep(3);
labeled['iq_features'] = labeled.image_quality.parallel_apply(lambda x: x.algorithm_features)
print('process iq_features_total'); time.sleep(3);
labeled['iq_features_total'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['total'])
print('process iq_features_low'); time.sleep(3);
labeled['iq_features_low'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['low'])
print('process iq_features_mid'); time.sleep(3);
labeled['iq_features_mid'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['mid'])
print('process iq_features_high'); time.sleep(3);
labeled['iq_features_high'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['high'])
print('process iq_features_sharpness'); time.sleep(3);
labeled['iq_features_sharpness'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['sharpness'])
print('process iq_features_smudge'); time.sleep(3);
labeled['iq_features_smudge'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['smudge'])
print('process iq_features_smudge_reason'); time.sleep(3);
labeled['iq_features_smudge_reason'] = labeled.iq_features.parallel_apply(lambda x: x['image_features']['smudge_reason'])
print('process iq_process_time'); time.sleep(3);
labeled['iq_process_time'] = labeled.image_quality.parallel_apply(lambda x: x.algorithm_process_time)
print('finish column mapping')

if 'iq_ground_truth' in labeled:
    labeled['binary_iq'] = labeled.iq.apply(lambda x: 'iq' if x != 'good' else 'non_iq')
    labeled['binary_iq_ground_truth'] = labeled.iq_ground_truth.apply(lambda x: 'iq' if x != 'good' else 'non_iq')

print(labeled.iq.value_counts())

labeled = labeled[['image_id', 'iq', 'iq_features', 'iq_features_total', 
                   'iq_features_low', 'iq_features_mid', 'iq_features_high', 
                   'iq_features_sharpness', 'iq_features_smudge', 'iq_features_smudge_reason', 'iq_process_time']]
labeled.to_csv(f'{save_directory}/iq_newmask.csv', index=False)
