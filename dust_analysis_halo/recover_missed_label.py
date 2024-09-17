import os
os.environ['BRT_ENV'] = 'prod'

aws_profile = 'default'
os.environ['AWS_PROFILE'] = aws_profile

from brtdevkit import setup_default_session
setup_default_session(aws_profile=aws_profile)

from brtdevkit.core import LabelPolicy
from brtdevkit.core import Annotation as AnnotationAPI
from brtdevkit.core import AnnotationBatch

import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from utils import normalize_image, get_sequences


def update_missed_label(label_dir, label_image_row, missed_label_image_id):
    # get label image 
    label_img = os.path.join(label_dir, label_image_row.annotation_pixelwise_0_save_path)
    
    values = {
        'kind': 'labelbox', # machine or labelbox
        'state': 'labeling', # state must be labeling in order to have the s3 information blank
        'style': 'pixelwise',
        'label_policy': '64a18d8843e37ee6fbff69ef',  # only label human
        'label_map_full': '64a18dd03e21af0a1b21ede7', # example label map id
        's3_key': '',
        's3_bucket': ''
    }
    annotation = AnnotationAPI(image_id=str(missed_label_image_id), values=values)
    annotation.create()
    
    # Upload the segmentation mask file from your local machine
    annotation.upload(label_img)
    
    # Update the state to ok now that the s3 location is set
    annotation = AnnotationAPI(
        image_id=str(annotation['image']), 
        values={'_id': str(annotation['id']), 'state': 'ok'}
    )
    annotation.update()


def recover_skipped_human_label(data_dir, seq_dfs, suffix, same_human_sequence):
    all_cameras = {'front': ['T01', 'T02', 'T03', 'T04'], 'right': ['T05', 'T06', 'T07', 'T08'], 'back': ['T09', 'T10', 'T11', 'T12'], 'left': ['T13', 'T14', 'T15', 'T16']}

    raw_ms_df = pd.read_csv(os.path.join(data_dir, 'master_annotations.csv'))
    raw_ms_df['camera_pair'] = raw_ms_df['unique_id'].apply(lambda s: s[-7:])
    labeled_ms_df = pd.read_csv(os.path.join(data_dir+suffix, 'master_annotations.csv'))
    labeled_ms_df.drop(columns=["label_counts"], inplace=True)
    labeled_ms_df['camera_pair'] = labeled_ms_df['unique_id'].apply(lambda s: s[-7:])
    print(raw_ms_df.shape, labeled_ms_df.shape)

    updated = set()
    for pod, seq_ids in same_human_sequence.items():
        for seq_id in seq_ids:
            # get seq df in pod
            seq_df = seq_dfs[seq_id]
            seq_df = seq_df[seq_df.camera_location.isin(all_cameras[pod])]
            # get corresponding raw seq df and labeled seq df
            raw_seq_df = raw_ms_df[raw_ms_df.id.isin(seq_df.id)]
            labeled_seq_df = labeled_ms_df[labeled_ms_df.id.isin(seq_df.id)]
            labeled_seq_df = labeled_seq_df.sort_values('collected_on')
            # get camera locations where there are human labels
            labeled_camera_pairs = labeled_seq_df.camera_pair.unique()
            for camera_pair in labeled_camera_pairs:
                raw_seq_cp_df = raw_seq_df[raw_seq_df.camera_pair == camera_pair]
                labeled_seq_cp_df = labeled_seq_df[labeled_seq_df.camera_pair == camera_pair]
                # assign label path to raw df
                for i, row in raw_seq_cp_df.iterrows():
                    labeled_rows = labeled_seq_cp_df[labeled_seq_cp_df.unique_id == row.unique_id]
                    if len(labeled_rows) > 0:
                        # print(pod, seq_id, row.id, 'has label')
                        pass
                    else:
                        # raw_seq_cp_df.loc[i, 'rectified_label_save_path'] = labeled_seq_cp_df.iloc[0].rectified_label_save_path
                        if not row.id in updated:
                            print(pod, seq_id, row.id, 'update label')
                            update_missed_label(data_dir+suffix, labeled_seq_cp_df.iloc[0], row.id)
                            updated.add(row.id)
            print(pod, seq_id, len(seq_df), len(raw_seq_df), len(labeled_seq_df), raw_seq_df.camera_pair.unique(), labeled_seq_df.camera_pair.unique())

    updated_df = pd.DataFrame(data={'id': list(updated)})
    print('updated image labels', updated_df.shape)
    updated_df.to_csv(os.path.join(data_dir, 'updated_skipped_human_label.csv'), index=False)
    
# # test updating a single image label
# root_dir = '/data/jupiter/li.yu/data/dust_data_colletion_for_july_1st'
# raw_dataset = 'halo_human_in_dust_day_collection_back_june05'
# labeled_dataset = 'halo_human_in_dust_day_collection_back_june05_human_labeled_stereo'

# labeled_df = pd.read_csv(os.path.join(root_dir, labeled_dataset, 'master_annotations.csv'))
# print(labeled_df.shape)

# data_dir = os.path.join(root_dir, raw_dataset)
# label_dir = os.path.join(root_dir, labeled_dataset)
# label_image_row = labeled_df[labeled_df.id == '6661214f167d5f2ee2089bd9'].iloc[0]
# missed_label_image_id = '6661269808707467156a4331'
# update_missed_label(label_dir, label_image_row, missed_label_image_id)


# batch update image labels
root_dir = '/data/jupiter/li.yu/data/dust_data_colletion_for_july_1st'
# raw_dataset = 'halo_human_in_dust_day_collection_may29'
raw_dataset = 'halo_human_in_dust_night_collection_june03_2'
# raw_dataset = 'halo_human_in_dust_day_collection_back_june05'
labeled_dataset = raw_dataset + '_human_labeled_stereo'

raw_df = pd.read_csv(os.path.join(root_dir, raw_dataset, 'master_annotations.csv'))
labeled_df = pd.read_csv(os.path.join(root_dir, labeled_dataset, 'master_annotations.csv'))
print(raw_df.shape, labeled_df.shape)
seq_dfs = get_sequences(raw_df, interval=60, per_camera_pair=False)  # break the data by intervals between sequences
print(raw_df.shape, len(seq_dfs))

# # recover skipped human labels in heavy dust, by images in the same sequence - halo_human_in_dust_day_collection_may29
# same_human_sequence = {'front': [0, 1, 3, 14], 'right': [11, 12, 13], 'back': [7, 8, 9, 10], 'left': [4, 5, 6]}
# suffix = '_human_labeled_stereo'
# recover_skipped_human_label(os.path.join(root_dir, raw_dataset), seq_dfs, suffix, same_human_sequence)

# recover skipped human labels in heavy dust, by images in the same sequence - halo_human_in_dust_night_collection_june03_2
same_human_sequence = {'front': [15, 17], 'right': [18], 'back': [7, 8, 9, 10], 'left': [5, 6]}
suffix = '_human_labeled_stereo'
recover_skipped_human_label(os.path.join(root_dir, raw_dataset), seq_dfs, suffix, same_human_sequence)

# # recover skipped human labels in heavy dust, by images in the same sequence - halo_human_in_dust_day_collection_back_june05
# same_human_sequence = {'back': [0, 1, 2, 3, 5, 6, 7]}
# suffix = '_human_labeled_stereo'
# recover_skipped_human_label(os.path.join(root_dir, raw_dataset), seq_dfs, suffix, same_human_sequence)