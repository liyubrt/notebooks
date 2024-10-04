# to download large image dataset and save in jpg format, change the following files
# note original files are ending with .bk, and changed files are ending with .save_jpg
# /home/li.yu/anaconda3/envs/pytorchlightning/lib/python3.8/site-packages/brtdevkit/data/core/dataset.py
# /home/li.yu/anaconda3/envs/pytorchlightning/lib/python3.8/site-packages/brtdevkit/util/aws/s3.py

import os
import sys
from brtdevkit.data import Dataset

""" # run this on cmd: 
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate brtdevkit
brt-devkit-auth
aws sso login --profile jupiter_prod_engineer-425642425116
"""

if len(sys.argv) > 1:
    i_list = sys.argv[1:]
for i in i_list:
    dataset_name = f'Jupiter_al_phase3_pool_pt{i}'
    # dataset_name = 'halo_sample_subset'
    # dataset_dir = os.path.join('/data/jupiter/datasets/dust_datasets', dataset_name)
    # dataset_dir = os.path.join('/data2/jupiter/datasets', dataset_name)
    dataset_dir = os.path.join('/data3/jupiter/datasets/large_datasets', dataset_name)
    # dataset_dir = os.path.join('/data/jupiter/li.yu/data', dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    test_dataset = Dataset.retrieve(name=dataset_name)
    test_df = test_dataset.to_dataframe()
    test_df = test_df[test_df.camera_location.str.endswith('left')]
    test_df.drop(columns=["artifact_raw_0_save_path"], inplace=True, errors="ignore")
    print(test_df.shape)
    test_dataset.download(dataset_dir, df=test_df, max_workers=32)