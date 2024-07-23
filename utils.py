import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


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


def get_sequences(df, interval=5*60, per_camera=False, per_camera_pair=False):
    df = df.sort_values('collected_on')
    if isinstance(df.iloc[0].collected_on, pd._libs.tslibs.timestamps.Timestamp):
        df['datetime'] = df.collected_on
    else:
        df['datetime'] = df.collected_on.apply(datetime.fromisoformat)
    sequence_dfs = []
    delta = timedelta(seconds=interval)
    start = True
    i0, i = 0, 0
    while i < len(df):
        if start:
            t0 = df.iloc[i].datetime
            start = False
        else:
            t1 = df.iloc[i].datetime
            if t1 - t0 > delta or i == len(df) - 1:
                chunk_df = df.iloc[i0 : i if i < len(df) - 1 else len(df)]
                if per_camera:
                    camera_locations = chunk_df.camera_location.unique()
                    camera_locations.sort()
                    for camera_location in camera_locations:
                        sequence_df = chunk_df[chunk_df.camera_location == camera_location]
                        sequence_df = sequence_df.sort_values('collected_on')
                        sequence_dfs.append(sequence_df)
                elif per_camera_pair:  # assume from master csv
                    chunk_df['camera_pair'] = chunk_df['unique_id'].apply(lambda s: s[-7:])
                    camera_pairs = chunk_df.camera_pair.unique()
                    camera_pairs.sort()
                    for camera_pair in camera_pairs:
                        sequence_df = chunk_df[chunk_df.camera_pair == camera_pair]
                        sequence_df = sequence_df.sort_values('collected_on')
                        sequence_dfs.append(sequence_df)
                else:
                    sequence_dfs.append(chunk_df)
                start = True
                i0 = i
            else:
                t0 = t1
        i += 1
    return sequence_dfs


def sample_per_min_per_camera(df, per_min_limit=30, per_camera=False):
    seq_dfs = get_sequences(df, interval=60, per_camera=per_camera)  # break the data by intervals between sequences
    print(df.shape, len(seq_dfs))
    selected_dfs = []
    for i, seq_df in enumerate(seq_dfs):
        time_span = (seq_df.iloc[-1].datetime - seq_df.iloc[0].datetime).total_seconds()
        limit_ids = int(time_span / 60 * per_min_limit)
        selected_dfs.append(seq_df.sample(min(len(seq_df), limit_ids)))
        print(i, seq_df.iloc[0].datetime, time_span, len(seq_df), limit_ids)
    selected_df = pd.concat(selected_dfs, ignore_index=True)
    print(df.shape, selected_df.shape)
    return selected_df


def plot_image(img, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_images(imgs, rows=1, cols=1, titles=[]):
    plt.figure(1, figsize=(8*cols, 4*rows))
    for i,img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()