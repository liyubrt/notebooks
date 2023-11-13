"""
Download corresponding Pack Perception artifacts from S3 given the S3 URI of master_annotations.csv.

Author: Rakhil Immidisetti, email: <rakhil.immidisetti@bluerivertech.com>
Copyright Blue River Technology Inc.
"""

import argparse
import os
import time
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
from brtdevkit.util.aws.s3 import S3, parallel_download


def prefetch_from_s3(master_csv_s3_uri, pack_perception_output_path, download_stereo=True, download_label=True):
    pack_perception_artifacts_to_download = []
    if download_stereo:
        pack_perception_artifacts_to_download.append("stereo_pipeline_npz_save_path")
    if download_label:
        pack_perception_artifacts_to_download.append("rectified_label_save_path")
    bucket = master_csv_s3_uri.split('/')[2]
    master_csv_s3_key = '/'.join(master_csv_s3_uri.split('/')[3:])
    master_csv_fname = os.path.basename(master_csv_s3_key)
    dataset_id = master_csv_fname.split('_')[0]
    pp_hash = os.path.basename(os.path.dirname(master_csv_s3_key))
    cache_s3_prefix = os.path.dirname(master_csv_s3_key)
    master_csv_write_path = os.path.join(pack_perception_output_path, master_csv_fname)
    dir_prefix = os.path.join(pack_perception_output_path, 'processed', 'images')
    print(f"Checking in cache : {cache_s3_prefix}")
    os.makedirs(pack_perception_output_path, exist_ok=True)
    
    # Log PP hash for tracking the config with which the artifacts have been generated
    with open(os.path.join(pack_perception_output_path, f'{dataset_id}.txt'), "w") as fp:
        fp.write(f'{pp_hash}\n')
    
    s3_client = S3()
    s3_client.download_file(bucket, master_csv_s3_key, master_csv_write_path)
    df = pd.read_csv(master_csv_write_path)
    print('Size of master csv:', len(df))
    
    # create all output folders
    df.id.apply(lambda row: Path(dir_prefix, row).mkdir(parents=True, exist_ok=True))

    files_to_download = pd.Series()
    for artifact in pack_perception_artifacts_to_download:
        if artifact in df.columns:
            files_to_download = files_to_download.append(
                pd.Series(
                    getattr(df, artifact).apply(
                        lambda artifact_path: (bucket, f"{cache_s3_prefix}/{artifact_path}")).values,
                    index=getattr(df, artifact).apply(
                        lambda artifact_path: Path(pack_perception_output_path, artifact_path)
                    )
                )
            )

    MAX_WORKERS = cpu_count() * 2  # no of processes to spin up to download
    print("Fetching files from S3 : ")
    start_time = time.time()
    parallel_download(files_to_download, max_workers=MAX_WORKERS)
    print(f"--- run time : {(time.time() - start_time)} seconds ---")
    
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Fetch mater_annotations.csv and PP artifacts from S3")
    # parser.add_argument("--master-csv-s3-uri",
    #                     type=str,
    #                     help="S3 uri of master_annotations.csv. "
    #                          "Eg: s3://mesa-states/prod/jupiter/pack_perception/ml/"
    #                          "f053a3adfe93b5e6a3a8d55f54602ec691165ab6_649de9ad4e873395e92ee11f95492756/"
    #                          "62a899ab50227481a1a2217b_master_annotations.csv",
    #                     required=True)
    # parser.add_argument("--output-path",
    #                     type=str,
    #                     help="Save fetched files in this dir. "
    #                          "Eg: /data/jupiter/datasets/humans_on_path_test_set_2022_vx_anno",
    #                     required=True)
    
    # args = parser.parse_args()

    # pp_hash, dataset_id, dataset = "5f3ec76b982566fd6e1a1ba6482330ce_b65750646597f0753af6db94c07446cf", "64ff41d09adbeff9ff6690f8", "mannequin_in_dust_v1"
    # pp_hash, dataset_id, dataset = "4f9098ace4422ddb23b5f77fa1a653c0_71573a7e6642901f3983f4b0d588b0c7", "652585580e4eef2185cbc19c",  "20230925_halo_rgb_stereo_train_v3_high_focal_loss"
    # pp_hash, dataset_id, dataset = "5f3ec76b982566fd6e1a1ba6482330ce_c7dcb27f46b4e853454683774b99bf3a", "652837ee74d48bca300372b9", "Jupiter_20230801_20231011_stop_event_labeling_candidates"
    pp_hash, dataset_id, dataset = "5f3ec76b982566fd6e1a1ba6482330ce_6317d7ad7d64bb0006522dd8284a78f6", "653a81a98d90e240199d31c0", "Jupiter_human_on_path_3_fn_sequence"
    master_csv_s3_uri = f"s3://blueriver-jupiter-data/pack_perception/ml/{pp_hash}/{dataset_id}_master_annotations.csv"
    # output_path = f"/data/jupiter/datasets/{dataset}"
    output_path = f"/data/jupiter/li.yu/data/{dataset}"

    prefetch_from_s3(master_csv_s3_uri, output_path, download_stereo=True, download_label=True)

    # rename master csv
    os.rename(os.path.join(output_path, f'{dataset_id}_master_annotations.csv'), os.path.join(output_path, 'master_annotations.csv'))
    