import os
import shutil
import numpy as np
import pandas as pd


root_dir = '/data2/jupiter/datasets/'
dataset = 'halo_rgb_stereo_train_v8_0'
mdf = pd.read_csv(os.path.join(root_dir, dataset, 'master_annotations.csv'))
adf = pd.read_csv(os.path.join(root_dir, dataset, 'annotations.csv'))
print(mdf.shape, adf.shape)

save_dir = '/data/jupiter/li.yu/data/halo_rgb_stereo_sample_train_2000'
n = 1000
sub_mdf = mdf.sample(n).drop_duplicates(subset='unique_id')
sub_adf = adf[(adf.id.isin(sub_mdf.id)) | (adf.id.isin(sub_mdf.id_right))].drop_duplicates(subset='id')
print(sub_mdf.shape, sub_adf.shape)

sub_mdf.to_csv(os.path.join(save_dir, 'master_annotations.csv'), index=False)
sub_adf.to_csv(os.path.join(save_dir, 'annotations.csv'), index=False)

image_dir = os.path.join(save_dir, 'images')
os.makedirs(image_dir, exist_ok=True)
for i,row in sub_adf.iterrows():
    shutil.copytree(os.path.join(root_dir, dataset, 'images', row.id), os.path.join(image_dir, row.id))

image_dir = os.path.join(save_dir, 'processed/images')
os.makedirs(image_dir, exist_ok=True)
for i,row in sub_mdf.iterrows():
    if not os.path.isdir(os.path.join(image_dir, row.id)):
        shutil.copytree(os.path.join(root_dir, dataset, 'processed/images', row.id), os.path.join(image_dir, row.id))