from genericpath import samefile
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


def normalize_image(image, hdr_mode=False, normalization_policy='percentile'):
    if image.dtype == np.uint8:
        normalization_policy = "default"
    if normalization_policy == "percentile" and hdr_mode:
        if image.dtype != np.float32 and image.dtype != np.uint32:
            raise ValueError('HDR image type is {} instead of float32 or uint32'.format(image.dtype))
        percentile_normalization_lower_bound = 0.1
        percentile_normalization_upper_bound = 99.5
        lower_bound = np.array([np.percentile(image[..., i], percentile_normalization_lower_bound, interpolation='lower')
                                for i in range(3)])
        upper_bound = np.array([np.percentile(image[..., i], percentile_normalization_upper_bound, interpolation='lower')
                                for i in range(3)])
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


root_dir = '/data/jupiter/datasets/Jupiter_train_v4_71'
data_dir = os.path.join(root_dir, 'images')
df = pd.read_csv(os.path.join(root_dir, 'master_annotations.csv'), low_memory=False)
print(df.shape)


# # build debayered image and label paths
# df = df.assign(image_save_path=lambda x: [str(os.path.join(root_dir, p)) for p in x.stereo_left_image])
# df = df.assign(label_save_path=lambda x: [str(os.path.join(root_dir, p)) for p in x.label_save_path])

# shape_dict = {'image_id':[], 'image_min':[], 'image_max':[], 'image_shape':[], 'label_shape':[], 'failed':[]}

# for i,sample_df in df.iterrows():
#     image_id = sample_df.image_id
#     shape_dict['image_id'].append(image_id)
#     try:
#         img = imageio.imread(sample_df.image_save_path)
#         lbl = imageio.imread(sample_df.label_save_path)
#         shape_dict['image_min'].append(img.min())
#         shape_dict['image_max'].append(img.max())
#         shape_dict['image_shape'].append(img.shape)
#         shape_dict['label_shape'].append(lbl.shape)
#         shape_dict['failed'].append('False')
#     except:
#         shape_dict['image_min'].append(np.nan)
#         shape_dict['image_max'].append(np.nan)
#         shape_dict['image_shape'].append(np.nan)
#         shape_dict['label_shape'].append(np.nan)
#         shape_dict['failed'].append('True')
#     if i % 2000 == 0:
#         print('processed', i)

# shape_df = pd.DataFrame(data=shape_dict)
# shape_df.to_csv('/data/jupiter/li.yu/data/Jupiter_train_v4_71/debayered_shape.csv', index=False)
# print(shape_df.shape)

# shape_df = pd.read_csv('/data/jupiter/li.yu/data/Jupiter_train_v4_71/debayered_shape.csv')
# shape_df = shape_df[shape_df.image_shape.str.endswith(', 3)')]
# sub_df = df[df.image_id.isin(shape_df.image_id)]
# print(sub_df.shape)
# print(df[~df.image_id.isin(shape_df.image_id)][['image_id', 'hdr_mode']].groupby('hdr_mode').count())
# print(sub_df[['image_id', 'hdr_mode']].groupby('hdr_mode').count())



# # check rectified image, label, raw image shapes
# shape_dict = {'image_id':[], 'image_shape':[], 'label_shape':[], 'raw_shape':[]}

# for i,sample_df in df.iterrows():
#     image_id = sample_df.image_id
#     data_path = os.path.join(root_dir, 'processed/images', image_id, 'stereo_output.npz')
#     img = np.load(data_path)['left']
#     label_path = os.path.join(root_dir, sample_df.rectified_label_save_path)
#     lbl = np.load(label_path)['left']
#     raw = cv2.imread(os.path.join(root_dir, sample_df.stereo_left_image))
#     shape_dict['image_id'].append(image_id)
#     shape_dict['image_shape'].append(img.shape[:2])
#     shape_dict['label_shape'].append(lbl.shape[:2])
#     shape_dict['raw_shape'].append(raw.shape[:2])
#     if i % 2000 == 0:
#         print('processed', i)

# shape_df = pd.DataFrame(data=shape_dict)
# shape_df.to_csv('/data/jupiter/li.yu/data/Jupiter_train_v4_71/shape.csv', index=False)
# print(shape_df.shape)

