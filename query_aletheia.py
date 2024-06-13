# to download large image dataset and save in jpg format, change the following files
# note original files are ending with .bk, and changed files are ending with .save_jpg
# /home/li.yu/anaconda3/envs/pytorchlightning/lib/python3.8/site-packages/brtdevkit/data/core/dataset.py
# /home/li.yu/anaconda3/envs/pytorchlightning/lib/python3.8/site-packages/brtdevkit/util/aws/s3.py

import os
from brtdevkit.data import Dataset

""" # run this on cmd: 
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate brtdevkit
brt-devkit-auth
aws sso login --profile jupiter_prod_engineer-425642425116
"""

dataset_name = 'halo_vehicles_in_dust_collection_june07_vehicle_labeled_stereo'
# dataset_name = 'halo_sample_subset'
dataset_dir = os.path.join('/data/jupiter/datasets', dataset_name)
# dataset_dir = os.path.join('/data/jupiter/li.yu/data', dataset_name)
os.makedirs(dataset_dir, exist_ok=True)

test_dataset = Dataset.retrieve(name=dataset_name)
test_df = test_dataset.to_dataframe()
test_dataset.download(dataset_dir, df=test_df, max_workers=8)