import os
import sys
import cv2
import json
import ndjson
import shutil
import random
import pickle
import imageio
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from pandarallel import pandarallel
logging.basicConfig(level=logging.INFO)


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


from skimage.exposure import match_histograms

def get_normed_rgb(df_row, data_dir):
    data_path = os.path.join(data_dir, df_row.stereo_pipeline_npz_save_path)
    img = np.load(data_path)['left']
    img_norm = normalize_image(img, True)
    return img_norm

def get_target_style_df(source_row, target_df, num_targets=1, sky_diff=0.1, implement_diff=0.2, value_diff=0.15):
    operation_time = source_row.operation_time
    camera_pod = source_row.camera_pod
    Sky_ratio = source_row.Sky_ratio
    Implement_ratio = source_row.Implement_ratio
    Value = source_row.Value
    if operation_time == 'unknown':
        target_style_df = target_df[(target_df.camera_pod == camera_pod) & 
                         (abs(target_df.Sky_ratio - Sky_ratio) <= sky_diff) & 
                         (abs(target_df.Implement_ratio - Implement_ratio) <= implement_diff) &
                         (abs(target_df.Value - Value) <= value_diff)]
    else:
        target_style_df = target_df[(target_df.camera_pod == camera_pod) & 
                         (target_df.operation_time == operation_time) & 
                         (abs(target_df.Sky_ratio - Sky_ratio) <= sky_diff) & 
                         (abs(target_df.Implement_ratio - Implement_ratio) <= implement_diff) &
                         (abs(target_df.Value - Value) <= value_diff)]
    if len(target_style_df) == 0:
        return None
    return target_style_df.sample(min(len(target_style_df), num_targets))

def color_transfer(source_row, source_datadir, target_df, target_datadir, save_dir, num_targets):
    # get filename and check if it exists already
    sample_save_dir = os.path.join(save_dir, source_row.id)
    color_transfer_npz = f"color_transfer_output_{source_row.unique_id[-7:]}.npz"
    sample_save_path = os.path.join(sample_save_dir, color_transfer_npz)
    if os.path.isfile(sample_save_path):
        return
    # compute new samples
    target_style_df = get_target_style_df(source_row, target_df, num_targets)
    if target_style_df is None:
        logging.info(source_row.unique_id)
        return
    sample = {"target_ids": []}
    for j,target_row in target_style_df.iterrows():
        simg = get_normed_rgb(source_row, source_datadir)
        timg = get_normed_rgb(target_row, target_datadir)
        matched = match_histograms(simg, timg, multichannel=True)
        sample["target_ids"].append(target_row.unique_id)
        sample[target_row.unique_id] = matched
    os.makedirs(sample_save_dir, exist_ok=True)
    np.savez_compressed(sample_save_path, **sample)
    return

def batch_color_transfer(source_df, source_datadir, target_df, target_datadir, save_dir, num_targets=1):
    # # do color transfer
    # pandarallel.initialize(nb_workers=8)
    # source_df = source_df.parallel_apply(lambda source_row: color_transfer(source_row, source_datadir, target_df, target_datadir, save_dir, num_targets), axis=1)
    
    # sanity check
    logging.info('perform sanity check')
    source_df['color_transfer_npz_save_path'] = ''
    good_unique_ids = []
    for i, source_row in source_df.iterrows():
        try:
            color_transfer_npz = f"color_transfer_output_{source_row.unique_id[-7:]}.npz"
            sample_save_path = os.path.join(save_dir, source_row.id, color_transfer_npz)
            source_df.at[i, 'color_transfer_npz_save_path'] = f'processed_color_transfer/images/{source_row.id}/{color_transfer_npz}'
            sample = np.load(sample_save_path)
            assert len(sample['target_ids']) > 0
            good_unique_ids.append(source_row.unique_id)
        except:
            logging.info(source_row.unique_id)
        if (i+1) % 1000 == 0:
            logging.info(f'processed {i+1} images')
    source_df = source_df[source_df.unique_id.isin(good_unique_ids)]
    source_df.to_csv(os.path.join(source_datadir, 'fl05_cc_master_annotations.csv'), index=False)



if __name__ == '__main__':

    root_dir = '/data/jupiter/datasets'

    # # Rev1 data
    # dataset1 = 'Jupiter_train_v5_11_20230508'
    # data_dir1 = os.path.join(root_dir, dataset1)
    # csv = os.path.join(data_dir1, 'master_annotations.csv')
    # df1 = pd.read_csv(csv)
    # print(df1.shape, flush=True)

    # RGBNir data
    dataset2 = 'Jupiter_halo_rgbnir_stereo_train_20230822_with_4_camera_implement_from_two_parts_subset_blur_dark_bright_label_proportion_planter_swapped_cams_reordered_cams_ocal_corrected_ocal_no_artifact_fix'
    data_dir2 = os.path.join(root_dir, dataset2)
    csv = os.path.join(data_dir2, 'fl05_master_annotations.csv')
    df2 = pd.read_csv(csv)
    print(df2.shape, flush=True)

    # Sample A data
    root_dir2 = '/data2/jupiter/datasets'
    dataset3 = '20230925_halo_rgb_stereo_train_v3'
    data_dir3 = os.path.join(root_dir2, dataset3)
    csv = os.path.join(data_dir3, 'master_annotations.csv')
    df3 = pd.read_csv(csv)
    print(df3.shape, flush=True)


    pods_to_cameras = {'front_pod': ['T01', 'T02', 'T03', 'T04'], 'right_pod': ['T05', 'T06', 'T07', 'T08'], 
                    'back_pod': ['T09', 'T10', 'T11', 'T12'], 'left_pod': ['T13', 'T14', 'T15', 'T16'], 
                    'implement_pod': ['I01', 'I02', 'I03', 'I04']}
    # get a dict with "T01_T03" : "front_pod"
    camera_pairs_to_pods = {cameras[i]+'_'+cameras[j]:pod for pod,cameras in pods_to_cameras.items() for i in range(len(cameras)) for j in range(i+1, len(cameras))}


    df2['camera_pair'] = df2['unique_id'].apply(lambda s: s[-7:])
    df2['camera_pod'] = df2['camera_pair'].apply(lambda s: camera_pairs_to_pods[s])
    # df2[['unique_id', 'operation_time', 'camera_pod']].groupby(['operation_time', 'camera_pod']).count()

    df3['camera_pair'] = df3['unique_id'].apply(lambda s: s[-7:])
    df3['camera_pod'] = df3['camera_pair'].apply(lambda s: camera_pairs_to_pods[s])
    # df3[['unique_id', 'operation_time', 'camera_pod']].groupby(['operation_time', 'camera_pod']).count()

    # read label count
    label_count_csv2 = '/data/jupiter/li.yu/exps/driveable_terrain_model/halo0822_8cls_0902/Jupiter_halo_rgbnir_stereo_train_20230822_with_4_camera_implement_from_two_parts_subset_blur_dark_bright_label_proportion_planter_swapped_cams_reordered_cams_ocal_corrected_ocal_no_artifact_fix/label_count.csv'
    label_count_df2 = pd.read_csv(label_count_csv2)
    label_count_csv3 = '/data/jupiter/li.yu/exps/driveable_terrain_model/rgb_baseline_sample_a_v3_2/20230925_halo_rgb_stereo_train_v3/label_count.csv'
    label_count_df3 = pd.read_csv(label_count_csv3)
    label_count_df2['Sky_ratio'] = label_count_df2['Sky'] / (512 * 640)
    label_count_df2['Implement_ratio'] = label_count_df2['Implement'] / (512 * 640)
    label_count_df3['Sky_ratio'] = label_count_df3['Sky'] / (512 * 640)
    label_count_df3['Implement_ratio'] = label_count_df3['Implement'] / (512 * 640)

    df2 = df2.merge(label_count_df2, on='unique_id')
    df3 = df3.merge(label_count_df3, on='unique_id')
    print(df2.shape, df3.shape, flush=True)


    # batch transfer rgbnir to a sample
    save_dir = os.path.join(root_dir, dataset2, 'processed_color_transfer/images')
    batch_color_transfer(df2, data_dir2, df3, data_dir3, save_dir, num_targets=1)

