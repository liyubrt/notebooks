import os
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
import matplotlib.pyplot as plt

DEFAULT_TONEMAP_PARAMS = {"policy": "tonemap", "alpha": 0.25, "beta": 0.9, "gamma": 0.9, "eps": 1e-6}
def normalize_image(image, hdr_mode=True, normalization_params=DEFAULT_TONEMAP_PARAMS, return_8_bit=False):
    """
    Normalize an 8 bit image according to the specified policy.
    If return_8_bit, this returns an np.uint8 image, otherwise it returns a floating point
    image with values in [0, 1].
    """
    normalization_policy = normalization_params['policy']
    lower_bound = 0
    upper_bound = 1
    if np.isnan(hdr_mode):
        hdr_mode = False

    if hdr_mode and image.dtype == np.uint8:
        # The image was normalized during pack-perception (tonemap)
        if return_8_bit:
            return image
        lower_bound = 0.0
        upper_bound = 255.0
    elif normalization_policy == "percentile" and hdr_mode:
        lower_bound = np.array([np.percentile(image[..., i],
                                              normalization_params['lower_bound'],
                                              interpolation='lower')
                                for i in range(3)])
        upper_bound = np.array([np.percentile(image[..., i],
                                              normalization_params['upper_bound'],
                                              interpolation='lower')
                                for i in range(3)])
    elif normalization_policy == "percentile_vpu" and hdr_mode:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        brightness = (3 * r + b + 4 * g) / 8
        lower_bound = np.percentile(brightness, normalization_params['lower_bound'],
                                    interpolation='lower')
        upper_bound = np.percentile(brightness, normalization_params['upper_bound'],
                                    interpolation='lower')
    elif normalization_policy == "3sigma" and hdr_mode:
        sigma_size = normalization_params['sigma_size']
        min_variance = normalization_params['min_variance']
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
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
    elif normalization_policy == 'tonemap' and hdr_mode:
        if image.dtype != np.float32 and image.dtype != np.uint32:
            raise ValueError('HDR image type is {} instead of float32 or uint32'.format(image.dtype))
        alpha = normalization_params.get('alpha', DEFAULT_TONEMAP_PARAMS['alpha'])
        beta = normalization_params.get('beta', DEFAULT_TONEMAP_PARAMS['beta'])
        gamma = normalization_params.get('gamma', DEFAULT_TONEMAP_PARAMS['gamma'])
        eps = normalization_params.get('eps', DEFAULT_TONEMAP_PARAMS['eps'])

        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        lum_in = 0.2126 * r + 0.7152 * g + 0.0722 * b
        lum_norm = np.exp(gamma * np.mean(np.log(lum_in + eps)))
        c = alpha * lum_in / lum_norm
        c_max = beta * np.max(c)
        lum_out = c / (1 + c) * (1 + c / (c_max ** 2))
        image = image * (lum_out / (lum_in + eps))[..., None]
    elif normalization_policy == "none" and hdr_mode:
        lower_bound = 0.0
        upper_bound = 2**20 - 1
    elif normalization_policy == "default" or not hdr_mode:
        assert np.max(image) <= 255 and np.min(image) >= 0, "Image with default " \
            "mode should be in range [0,255]"
        lower_bound = 0.0
        upper_bound = 255.0
    else:
        raise ValueError(
            f"--normalization-policy '{normalization_policy}' is not supported! "
            f"(on image with hdr_mode={hdr_mode})")

    image = (image.astype(np.float32, copy=False) - lower_bound) / (upper_bound - lower_bound)

    if return_8_bit:
        image = np.clip(image * 255.0, 0.0, 255.0)
        image = np.uint8(image)
    else:
        image = np.clip(image, 0.0, 1.0)

    return image


data_root_dir = '/data/jupiter/li.yu/data'
unlabeled_datasets = ["Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2", 
                      "Jupiter_2023_04_05_loamy869_dust_collection_stereo", 
                      "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo"]
labeled_datasets = ["Jupiter_2023_03_02_and_2930_human_vehicle_in_dust_labeled", 
                    "Jupiter_2023_March_29th30th_human_vehicle_in_dust_front_pod_labeled", 
                    "Jupiter_2023_04_05_loamy869_dust_collection_stereo_labeled", 
                    "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo_labeled"]
pred_root = '/data/jupiter/li.yu/exps/driveable_terrain_model/'

i = 0

# os.makedirs(os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_images'), exist_ok=True)
# master_df = pd.read_csv(os.path.join(data_root_dir, unlabeled_datasets[i], 'master_annotations.csv'), low_memory=False)
# print(master_df.shape)
# for idx, row in master_df.iterrows():
#     data_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'processed/images', row.id, 'stereo_output.npz')
#     img = np.load(data_path)['left']
#     img_norm = normalize_image(img, row.hdr_mode)
#     img_norm = (img_norm * 255).astype(np.uint8)
#     img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
#     save_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_images', row.id+'.png')
#     cv2.imwrite(save_path, img_norm)
#     if idx % 1000 == 0:
#         print(f'saved {idx+1} images')

os.makedirs(os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_labels'), exist_ok=True)
master_df = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i], 'master_annotations.csv'), low_memory=False)
print(master_df.shape)
for idx, row in master_df.iterrows():
    label_path = os.path.join(data_root_dir, labeled_datasets[i], row.rectified_label_save_path)
    lbl = np.load(label_path)['left']
    save_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_labels', row.id+'.png')
    cv2.imwrite(save_path, lbl)
    if idx % 1000 == 0:
        print(f'saved {idx+1} images')

master_df = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i+1], 'master_annotations.csv'), low_memory=False)
print(master_df.shape)
for idx, row in master_df.iterrows():
    label_path = os.path.join(data_root_dir, labeled_datasets[i+1], row.rectified_label_save_path)
    lbl = np.load(label_path)['left']
    save_path = os.path.join(data_root_dir, unlabeled_datasets[i], 'normed_labels', row.id+'.png')
    cv2.imwrite(save_path, lbl)
    if idx % 1000 == 0:
        print(f'saved {idx+1} images')