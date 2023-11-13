import os
import cv2
import json
import shutil
import random
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

def normalize_image(image, hdr_mode=False, normalization_policy='percentile'):
    if image.dtype == np.uint8:
        normalization_policy = "default"
    if "percentile" in normalization_policy and hdr_mode:
        if image.dtype != np.float32 and image.dtype != np.uint32:
            raise ValueError('HDR image type is {} instead of float32 or uint32'.format(image.dtype))
        percentile_normalization_lower_bound = 0.1
        percentile_normalization_upper_bound = 99.5

        if normalization_policy == "percentile":
            lower_bound = np.array([np.percentile(image[..., i],
                                                  percentile_normalization_lower_bound,
                                                  interpolation='lower')
                                    for i in range(3)])
            upper_bound = np.array([np.percentile(image[..., i],
                                                  percentile_normalization_upper_bound,
                                                  interpolation='lower')
                                    for i in range(3)])
        elif normalization_policy == "percentile_vpu":
            r, g, b = image[..., 0], image[..., 1], image[..., 2]
            brightness = (3 * r + b + 4 * g) / 8
            lower_bound = np.percentile(brightness, percentile_normalization_lower_bound,
                                        interpolation='lower')
            upper_bound = np.percentile(brightness, percentile_normalization_upper_bound,
                                        interpolation='lower')

        image = (image.astype(np.float32) - lower_bound) / (upper_bound - lower_bound)
        image = np.clip(image, 0.0, 1.0)
    elif normalization_policy == "3sigma" and hdr_mode:
        if image.dtype != np.float32 and image.dtype != np.uint32:
            raise ValueError('HDR image type is {} instead of float32 or uint32'.format(image.dtype))
        sigma_size = 3
        min_variance = 1200
        r, g, b = image[...,0], image[...,1], image[...,2]
        brightness = (3 * r + b + 4 * g) / 8
        mean, sigma = np.mean(brightness), np.std(brightness)
        brightness_min, brightness_max = np.min(brightness), np.max(brightness)
        if (sigma * sigma_size) > mean:
            lmin = brightness_min
            lmax = min(brightness_max, mean * sigma_size)
            if (lmax - lmin) < min_variance:
                lmax = lmin + min_variance
            lower_bound = lmin
            upper_bound = lmax
        else:
            mean_var = mean - sigma_size * sigma
            output_min = max(brightness_min, mean_var)
            mean_var = mean + sigma_size * sigma
            output_max = min(brightness_max, mean_var)
            if (output_max - output_min) < min_variance:
                output_min = mean - min_variance / 2.0
                output_min = 0 if output_min < 0 else output_min
                output_max = output_min + min_variance
            lower_bound = output_min
            upper_bound = output_max
        image = (image.astype(np.float32) - lower_bound) / (upper_bound - lower_bound)
        image = np.clip(image, 0.0, 1.0)
    elif normalization_policy == 'tonemap' and hdr_mode:
        if image.dtype != np.float32 and image.dtype != np.uint32:
            raise ValueError('HDR image type is {} instead of float32 or uint32'.format(image.dtype))
        alpha = 0.5
        r, g, b = image[...,0], image[...,1], image[...,2]
        L_in = 0.27 * r + 0.67 * g + 0.06 * b
        Lw = np.exp(np.mean(np.log(L_in + 1e-6)))
        n = alpha * L_in / Lw
        L_out = n / (1 + n) * (1 + n / (n.max() ** 2))
        image = np.clip(image.astype(np.float32) * (L_out / L_in)[..., None], 0.0, 1.0)
    elif normalization_policy == "none" and hdr_mode:
        lower_bound = 0.0
        upper_bound = 2**20 - 1
        image = (image.astype(np.float32) - lower_bound) / (upper_bound - lower_bound)
        image = np.clip(image, 0.0, 1.0)
    elif normalization_policy == "default" or not hdr_mode:
        assert np.max(image) <= 255 and np.min(image) >= 0, \
            "Image with linear model should be in range [0,255]"
        lower_bound = 0.0
        upper_bound = 255.0
        image = (image.astype(np.float32) - lower_bound) / (upper_bound - lower_bound)
        image = np.clip(image, 0.0, 1.0)
    else:
        raise ValueError("normalization_policy is not supported!")
    return image



root_dir = '/data/jupiter/avinash.raju/iq_2022_v8_anno'
# root_dir = '/data/jupiter/datasets/dust_test_2022_v4_anno_HEAVY_DUST'
data_dir = os.path.join(root_dir, 'images')
df = pd.read_csv(os.path.join(root_dir, 'master_annotations.csv'), low_memory=False)
df = df.sort_values('id')
# df = pd.read_csv(os.path.join(root_dir, 'master_annotations_labeled.csv'), low_memory=False)
df.shape


# IQ result
iq_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/v51rd_7cls_imgaug_highbz_100epoch_0121_dustv51_0130/iq_2022_v8_anno/iq_2022_v8_anno_iq.csv'
iq_df = pd.read_csv(iq_csv)
# iq_df = pd.read_csv(os.path.join(root_dir, 'iq_newmask.csv'))
# iq_df['iq'] = iq_df['iq_status']
iq_df = iq_df.sort_values('id')
print(iq_df.shape, '# bad iq from IQ algorithm:', iq_df[iq_df.iq != 'good'].shape)
# iq_df.groupby('iq').count()

# segmentation result
# new_seg_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/471_cloud_v45_cutnpaste_s35/iq_2022_v8_anno/output.csv'
# new_seg_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/471_local_v55_cutout_daware_w10_s35/iq_2022_v8_anno/output.csv.percentile_vpu'
# new_seg_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/471_local_v55_cutout_daware_w10_s35/dust_test_2022_v4_anno_HEAVY_DUST/output.csv.labeled.percentile_vpu'
# new_seg_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/dust_471_1e3_v083_484_local_tonemap_nonocclu_debris_birds_s45/dust_test_2022_v4_anno_HEAVY_DUST_epoch10.seg/output.csv.labeled'
# new_seg_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/human_detector_2C_v471_rear_dustHead_percentile_brightness_20220809-1/iq_2022_v8_anno.percentile_vpu/output.csv.seg'
new_seg_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/v51rd_7cls_imgaug_highbz_100epoch_0121_dustv51_0130/iq_2022_v8_anno/output.csv'
new_seg_df = pd.read_csv(new_seg_csv)
new_seg_df = new_seg_df.sort_values('id')
print(new_seg_df.shape, '# seg FPs:', new_seg_df[new_seg_df.state == 'false_positive'].shape)

# dust detection result
# new_dust_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/v471_6cls_base_dusthead_0727/iq_2022_v8_anno/output.csv'
# new_dust_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/v471_6cls_base_dusthead_0812/iq_2022_v8_anno/output.csv.percentile_vpu'
# new_dust_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/v471_6cls_base_dusthead_0812/dust_test_2022_v4_anno_HEAVY_DUST/output.csv.labeled.percentile_vpu'
# new_dust_csv = '/mnt/sandbox1/rakhil.immidisetti/output/job_quality/dust_471_1e3_v083_484_local_tonemap_nonocclu_debris_birds_s45/dust_test_2022_v4_anno_HEAVY_DUST_labeled_10_epoch_model/output.csv'
# new_dust_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/human_detector_2C_v471_rear_dustHead_percentile_brightness_20220809-1/iq_2022_v8_anno.percentile_vpu/output.csv.dust'
new_dust_csv = '/data/jupiter/li.yu/exps/driveable_terrain_model/v51rd_7cls_imgaug_highbz_100epoch_0121_dustv51_0130/iq_2022_v8_anno_dust/output.csv'
new_dust_df = pd.read_csv(new_dust_csv)
new_dust_df = new_dust_df.sort_values('id')
dust_thres = 0.15 * 100
print(new_dust_df.shape, f'# dust FPs under threshold {dust_thres}:', df[new_dust_df.pred_dust_ratio > dust_thres].shape)


# combined results
print()
print('# IQ bad or seg FPs:', df[(iq_df.iq != 'good') | (new_seg_df.state == 'false_positive')].shape)
print('# seg FPs or dust FPs:', df[(new_seg_df.state == 'false_positive') | (new_dust_df.pred_dust_ratio > dust_thres)].shape)
fp_df = df[(iq_df.iq != 'good') | (new_seg_df.state == 'false_positive') | (new_dust_df.pred_dust_ratio > dust_thres)]
print('# IQ bad or seg FPs or dust FPs:', fp_df.shape)

# calculate recall
fn_df = df[~df.image_id.isin(fp_df.image_id)]
print(fp_df.shape, fn_df.shape)
print(fn_df.sample(5).id.to_list())
print('Recall:', fp_df.shape[0] / df.shape[0])
# print('Recall:', df[(df.image_id.isin(fp_df.image_id)) & (new_dust_df.gt_dust_label > dust_thres)].shape[0] / new_dust_df[new_dust_df.gt_dust_label > dust_thres].shape[0])
