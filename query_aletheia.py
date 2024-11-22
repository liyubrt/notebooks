# to download large image dataset and save in jpg format, change the following files
# note original files are ending with .bk, and changed files are ending with .save_jpg
# /home/li.yu/anaconda3/envs/pytorchlightning/lib/python3.8/site-packages/brtdevkit/data/core/dataset.py
# /home/li.yu/anaconda3/envs/pytorchlightning/lib/python3.8/site-packages/brtdevkit/util/aws/s3.py

import os
import sys
from brtdevkit.data import Dataset

""" # run this on cmd: 
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate query
brt-devkit-auth
aws sso login --profile jupiter_prod_engineer-425642425116
"""

if len(sys.argv) > 1:
    i_list = sys.argv[1:]
for i in i_list:
    # dataset_name = f'Jupiter_al_phase3_pool_pt{i}'
    dataset_name = i
    print(f'downloading {dataset_name}')
    # dataset_name = 'halo_sample_subset'
    # dataset_dir = os.path.join('/data/jupiter/datasets/dust_datasets', dataset_name)
    # dataset_dir = os.path.join('/data2/jupiter/datasets', dataset_name)
    dataset_dir = os.path.join('/data3/jupiter/datasets/model_positive', dataset_name)
    # dataset_dir = os.path.join('/data/jupiter/li.yu/data', dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # if os.path.isfile(os.path.join(dataset_dir, 'annotations.csv')):
    #     continue
    
    test_dataset = Dataset.retrieve(name=dataset_name)
    test_df = test_dataset.to_dataframe()
    # test_df = test_df[test_df.camera_location.str.endswith('left')]
    # test_df.drop(columns=["artifact_raw_0_save_path"], inplace=True, errors="ignore")
    print(test_df.shape)
    test_dataset.download(dataset_dir, df=test_df, max_workers=16)



# from jupiterdata.db import databricks_connector
# databricks = databricks_connector.Databricks()
# # note on json extraction https://docs.databricks.com/en/sql/language-manual/sql-ref-json-path-expression.html
# df = databricks.execute(query)
# print(df.shape, len(df)//2)

  # IMJ.has_human_annotation,
  # IMJ.has_nearby_stop_event,
  # IMJ.operating_field_name,
  # IMJ.farm,
  # IMJ.bag_name,
  # IMJ.closest_object_info__json,
  # IMJ.created_at,
  # IMJ.hard_drive_name,
  # IMJ.state,
  # IMJ.hdr_mode,
  # IMJ.group_id,
  # IMJ.jdb_s3_path,
  # IMJ.ros_s3_path,
  # IMJ.bundle,
  # IMJ.gps_can_data__json,
  # IMJ.implement_angle_data__json,
  # IMJ.autonomy_state__json,
  # IMJ.sensor_type,
  # IMJ.geohash,
  # IMJ.calibration_data__json,
  # IMJ.tractor_type
  # IMJ.robot_name = 'bedrock_411'
  # AND IMJ.hard_drive_name = 'JUPD-0612_2022-1-3'
  # AND IMJ.collected_on > date('2024-04-22')
  # AND IMJ.collected_on <= date('2024-04-24')

# query = """
# SELECT id, image, properties__json FROM annotation_jupiter
# WHERE style = 'categorical'
# AND created_at > date('2024-01-01')
# AND state in ('review', 'ok')
# AND properties__json LIKE '%"title": "human\_pose"%'
# AND kind = 'labelbox'
# """

# # query for categorical labels
# query = f"""
# SELECT image AS id, properties__json, created_at FROM annotation_jupiter
# WHERE style = 'categorical'
# AND state in ('review', 'ok')
# AND label_map_full = '66343128ce27e38fac43fc3e'
# AND kind = 'labelbox'
# AND annotation_jupiter.vendor_metadata__json LIKE '%vendor_project_id": "clvzvij3v02w1070x7p0hdc55%'
# """

# query = f"""
# SELECT ANJ.id as id, ANJ.image as image, ANJ.properties__json as properties__json, ANJ.created_at as created_at
# FROM mesa_prod.mesa_lake_prod.annotation_jupiter as ANJ
# WHERE ANJ.style = 'categorical'
# AND ANJ.created_at > date('2024-01-01')
# AND ANJ.state in ('review', 'ok')
# AND (properties__json LIKE '%"title": "Pose"%' OR properties__json LIKE '%"title": "Human Clothing"%' OR properties__json LIKE '%"title": "Human Occlusion"%') 
# AND (properties__json LIKE '%"title": "human_pose"%' OR properties__json LIKE '%"title": "human_clothing"%' OR properties__json LIKE '%"title": "human_occlusion"%') 
# AND ANJ.kind = 'labelbox'
#   hard_drive_name = 'JUPD-0325_2024-3-19'
#   AND gps_can_data__json IS NOT NULL
#   AND special_notes IS NOT NULL
#   AND SUBSTRING(IMJ.geohash, 1, 6) IN {tuple(geohash6_test_list)}
#   AND IMJ.sensor_type = 'VD6763'
#   AND (IMJ.jdb_s3_path like '%.mcap' OR IMJ.calibration_data__json:serial_number LIKE 'PCE7T_B%')
# """