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
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
from pandarallel import pandarallel
import matplotlib.pyplot as plt
sys.path.append('../')
from utils import normalize_image, plot_image, plot_images

root_dir = '/data3/jupiter/datasets/public_datasets/OpenImages/'

# load label csv
split = 'train'  # train, validation, test
label_csv = os.path.join(root_dir, 'meta_data', f'oidv7-{split}-annotations-human-imagelabels.csv')
image_ids = os.listdir(os.path.join(root_dir, split))
image_ids = [f[:-4] for f in image_ids]
np.savetxt(os.path.join(root_dir, 'meta_data', f'{split}_saved.txt'), image_ids, delimiter='\n', fmt='%s')
# image_ids = list(np.loadtxt(os.path.join(root_dir, 'meta_data', f'{split}_saved.txt'), dtype=str))
df = pd.read_csv(label_csv)
print('# saved images', len(image_ids), 'loaded from csv', df.shape)

# load class description
description_csv = os.path.join(root_dir, 'meta_data', 'oidv7-class-descriptions.csv')
description_df = pd.read_csv(description_csv)
description_dict = dict(zip(description_df.LabelName, description_df.DisplayName))
print(description_df.shape, len(description_dict))

# load trainable classes
trainable_txt = os.path.join(root_dir, 'meta_data', 'oidv7-classes-trainable.txt')
with open(trainable_txt, 'r') as f:
    trainable_class_ids = f.readlines()
    trainable_class_ids = {s[:-1] for s in trainable_class_ids}
print('trainable classes', len(trainable_class_ids))

# load 600 boxable classes
boxable_classes_csv = os.path.join(root_dir, 'meta_data', 'oidv7-class-descriptions-boxable.csv')
boxable_classes_df = pd.read_csv(boxable_classes_csv)
print('boxable classes', boxable_classes_df.shape)

# merge class description and trainable info
df = df.merge(description_df, on='LabelName', how='left')
df['Trainable'] = df['LabelName'].apply(lambda l: True if l in trainable_class_ids else False)
print(df.shape, len(df.ImageID.unique()))

# get trainable and boxable df
df_use = df[(df.DisplayName.isin(boxable_classes_df.DisplayName)) & (df.Trainable == True) & (df.Confidence > 0.0) & (df.ImageID.isin(image_ids))]
print(df_use.shape, len(df_use.ImageID.unique()))


# convert classes to binary labels in a row
df_use2 = pd.DataFrame(data={'ImageID': list(df_use.ImageID.unique())})
print(df_use2.shape)

# group by image id
g = dict(iter(df_use.groupby('ImageID')))
print(len(g))

# get boxable class list
boxable_classes = boxable_classes_df.DisplayName.to_list()
print(len(boxable_classes))

def get_binary_label(row, g, boxable_classes):
    names = g[row.ImageID].DisplayName.to_list()
    for c in boxable_classes:
        row[c] = 1.0 if c in names else 0.0
    return row

pandarallel.initialize(nb_workers=32, progress_bar=True)
df_use2 = df_use2.parallel_apply(lambda row: get_binary_label(row, g, boxable_classes), axis=1)
print(df_use2.shape)
df_use2.to_csv(os.path.join(root_dir, 'meta_data', f'{split}_600classes.csv'), index=False)