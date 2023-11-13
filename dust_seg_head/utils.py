import os
import sys
import cv2
import time
import random
import pickle
import shutil
import imageio
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import AnyStr, Optional, Tuple


def normalize_image(image, hdr_mode=False, normalization_policy='default'):
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

def normalize_and_clip_depth(depth, max_depth):
    """
    Return an optionally normalized (and clipped) depth.
    """
    depth[np.isnan(depth)] = max_depth
    depth[depth > max_depth] = max_depth
    depth = ((depth) / max_depth).astype(np.float32)
    return depth

def depth_from_point_cloud(point_cloud: np.ndarray,
                            clip_and_normalize: Optional[bool] = True,
                            max_depth: Optional[float] = 100.0,
                            make_3d: Optional[bool] = True) -> np.ndarray:
    """
    Given a np ndarray of point_cloud in channel-order [x, y, z], this function returns the z-channel after
    appropriately transforming it
    """
    depth = point_cloud[..., -1]
    if clip_and_normalize:
        depth = normalize_and_clip_depth(depth, max_depth)
    if make_3d:
        return np.atleast_3d(depth)
    return depth

def get_image_and_depth_from_npz_file(npz_path: AnyStr,
                                      clip_and_normalize: Optional[bool] = True,
                                      max_depth: Optional[float] = 100.0,
                                      make_3d: Optional[bool] = True,
                                      hdr_mode: Optional[bool] = False,
                                      normalization_policy='percentile') -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to read image and depth from an npz file that gets output from pack_perception_stack_data.py
    """
    if not os.path.isfile(npz_path):
        raise ValueError(f'Given npz_path {npz_path} doesnt exist')
    try:
        stereo_data = np.load(npz_path)
        image = stereo_data['left']
        image = normalize_image(image, hdr_mode, normalization_policy)
        depth = depth_from_point_cloud(stereo_data['point_cloud'], clip_and_normalize, max_depth, make_3d)
    except Exception as e:
        raise ValueError('Failed loading {} with error: \n {}'.format(npz_path, e))
    return image, depth


def get_tire_mask(image_path, input_size=[1024, 512]):
    if not os.path.isfile(image_path):
        return None
    try:
        mask_image = imageio.imread(
            image_path, as_gray=True
        )  # should read out an image with values == 0 or 255
        mask = mask_image > 0
        return cv2.resize(mask.astype(np.uint8), input_size, cv2.INTER_NEAREST_EXACT).astype(np.bool)
    except Exception as e:
        raise ("Error reading tire mask:", e)