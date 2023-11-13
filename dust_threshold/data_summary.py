import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use('Agg')
# from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from pprint import pprint


def merge_meta_with_pred(master_df, label_df, unlabeled_pred_df, labeled_pred_df):
    print('master_df', master_df.shape)
    df = master_df[['id', 'collected_on', 'camera_location', 'operation_time', 'geohash']]
    
    # merge with label_df
    df = df.merge(label_df, on='id', how='left')
    df = df.fillna(0)
    print('merge label counts', df.shape)

    # load prediction on unlabeled data to get dust ratios
    print('unlabeled_pred_df', unlabeled_pred_df.shape)
    if not 'total_averaged_dust_ratio' in unlabeled_pred_df:
        unlabeled_pred_df['total_averaged_dust_ratio'] = unlabeled_pred_df['total_averaged_dust_conf']
        unlabeled_pred_df['triangle_averaged_dust_ratio'] = unlabeled_pred_df['masked_avg_dust_conf']
    df = df.merge(unlabeled_pred_df[['id', 'total_averaged_dust_ratio', 'triangle_averaged_dust_ratio']], on='id')
    print('merge unlabeled dust ratios', df.shape)

    # load prediction on labeled data to get the prediction "state"
    print('labeled_pred_df', labeled_pred_df.shape)
    # convert LO states to regular states and fill empty states with TNs
    df = df.merge(labeled_pred_df[['id', 'state']], on='id', how='left').drop_duplicates(subset=['id'])
    df = df.fillna('true_negative')
    df = df.replace('large_object_true_positive', 'true_positive')
    df = df.replace('large_object_false_negative', 'false_negative')
    print('merge labeled states', df.shape)

    # sort by time and add datetime column
    df = df.sort_values('collected_on')
    df['datetime'] = df.collected_on.apply(datetime.fromisoformat)
    df['datehm'] = df.collected_on.apply(lambda x:str(x)[:16])
    print('final_df', df.shape)
    print('# TPs', len(df[df.state == 'true_positive']), '# Positives', len(df[(df.state == 'true_positive') | (df.state == 'false_negative')]))
    
    return df


data_root_dir = '/data/jupiter/li.yu/data'
unlabeled_datasets = ["Jupiter_2023_03_29_10pm_30_3pm_Loamy_812_stops_stereo_2", 
                      "Jupiter_2023_04_05_loamy869_dust_collection_stereo", 
                      "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo"]
labeled_datasets = ["Jupiter_2023_03_02_and_2930_human_vehicle_in_dust_labeled", 
                    "Jupiter_2023_March_29th30th_human_vehicle_in_dust_front_pod_labeled", 
                    "Jupiter_2023_04_05_loamy869_dust_collection_stereo_labeled", 
                    "Jupiter_2023_may_loamy731_vehicle_dust_human_stereo_labeled"]
pred_root = '/data/jupiter/li.yu/exps/driveable_terrain_model/'
train_id = 'v57rd_4cls_tiny0occluded5reverse5triangle5_msml_0305'


# set 1
i = 0
master_df = pd.read_csv(os.path.join(data_root_dir, unlabeled_datasets[i], 'master_annotations.csv'), low_memory=False)
label_df1 = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i], 'label_count.csv'))
label_df2 = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i+1], 'label_count.csv'))
label_df = pd.concat([label_df1, label_df2], ignore_index=True)
unlabeled_pred_df = pd.read_csv(os.path.join(pred_root, train_id, unlabeled_datasets[i]+'_epoch43_newmask', 'dust_ratio.csv'))
labeled_pred_df1 = pd.read_csv(os.path.join(pred_root, train_id, labeled_datasets[i]+'_epoch43', 'output.csv'))
labeled_pred_df2 = pd.read_csv(os.path.join(pred_root, train_id, labeled_datasets[i+1]+'_epoch43', 'output.csv'))
labeled_pred_df = pd.concat([labeled_pred_df1, labeled_pred_df2], ignore_index=True)
df1 = merge_meta_with_pred(master_df, label_df, unlabeled_pred_df, labeled_pred_df)

# set 2
i = 1
master_df = pd.read_csv(os.path.join(data_root_dir, unlabeled_datasets[i], 'master_annotations.csv'), low_memory=False)
label_df = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i+1], 'label_count.csv'))
unlabeled_pred_df = pd.read_csv(os.path.join(pred_root, train_id, unlabeled_datasets[i]+'_epoch43', 'dust_ratio.csv'))
labeled_pred_df = pd.read_csv(os.path.join(pred_root, train_id, labeled_datasets[i+1]+'_epoch43', 'output.csv'))
df2 = merge_meta_with_pred(master_df, label_df, unlabeled_pred_df, labeled_pred_df)

# set 3
i = 2
master_df = pd.read_csv(os.path.join(data_root_dir, unlabeled_datasets[i], 'master_annotations.csv'), low_memory=False)
label_df = pd.read_csv(os.path.join(data_root_dir, labeled_datasets[i+1], 'label_count.csv'))
unlabeled_pred_df = pd.read_csv(os.path.join(pred_root, train_id, unlabeled_datasets[i]+'_epoch43', 'dust_ratio.csv'))
labeled_pred_df = pd.read_csv(os.path.join(pred_root, train_id, labeled_datasets[i+1]+'_epoch43', 'output.csv'))
df3 = merge_meta_with_pred(master_df, label_df, unlabeled_pred_df, labeled_pred_df)

df = pd.concat([df1, df2, df3], ignore_index=True)
print('merge all three sets (double df3 as it is halved) to get df', df.shape)
pdf = df[(df.state == 'true_positive') | (df.state == 'false_negative')]
print('positive images', pdf.shape)

# save dfs
df1.to_csv(os.path.join(pred_root, train_id, unlabeled_datasets[0]+'.csv'), index=False)
df2.to_csv(os.path.join(pred_root, train_id, unlabeled_datasets[1]+'.csv'), index=False)
df3.to_csv(os.path.join(pred_root, train_id, unlabeled_datasets[2]+'.csv'), index=False)
