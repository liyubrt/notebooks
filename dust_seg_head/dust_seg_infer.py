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
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim

from brt_model import BrtResnetPyramidLite12, freeze_encoder, load_states
from utils import normalize_image


class DustData(torch.utils.data.Dataset):
    def __init__(self, data_dir, df, transform, mode, 
                 rgb='debayered', normalization_policy='tonemap'):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        self.mode = mode
        self.rgb = rgb
        self.normalization_policy = normalization_policy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_df = self.df.iloc[idx]
        image_id = sample_df.id
        
        try:
            if self.rgb == 'debayered':
                img_path = os.path.join(self.data_dir, sample_df['artifact_debayeredrgb_0_save_path'])
                image = imageio.imread(img_path)
            elif self.rgb == 'rectified':
                npz_path = os.path.join(self.data_dir, sample_df['stereo_pipeline_npz_save_path'])
                stereo_data = np.load(npz_path)
                image = stereo_data['left']
                hdr_mode = sample_df['hdr_mode'] if 'hdr_mode' in sample_df and ~np.isnan(sample_df['hdr_mode']) else False
                image = normalize_image(image, hdr_mode, self.normalization_policy)
            else:
                logging.warning(f'rgb mode is not correctly specified: {self.rgb}')
        except:
            logging.warning(f'{image_id} has corrupted rgb file')
            image = np.zeros((512, 1024, 3), dtype=np.uint8)
        
        if self.mode == 'train' and random.random() > 0.5:
            image = np.fliplr(image)
        
        image = image.transpose((2, 0, 1)).copy()
        image = self.transform(torch.from_numpy(image) / 255.)
        return image_id, image


class Solver:
    def __init__(self, testloader, seg_model_path):
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.load_model('brt', seg_model_path)  # resnet18 or brt
        print('# tunable params', sum(p.numel() for p in net.parameters() if p.requires_grad))
        net = nn.DataParallel(net)
        self.net = net.to(self.device)
        self.net.eval()

    def load_model(self, model_name, seg_model_path):
        net = None
        if model_name.startswith('resnet'):  # load resnet model
            assert model_name == 'resnet18' or model_name == 'resnet50'
            if model_name == 'resnet18':
                net = models.resnet18(pretrained=True)
            elif model_name == 'resnet50':
                net = models.resnet50(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, 2)  # reset final fully connected layer
        elif model_name.startswith('brt'):  # load brt model
            params = {"input_dims": 3, "num_classes": 2, "seg_output": True, "cls_output": False, "add_softmax_layer": True,
                      "model_params": {"num_block_layers": 2, "widening_factor": 2, "upsample_mode": "nearest"}}
            net = BrtResnetPyramidLite12(params)
            print('loading seg model states from', seg_model_path)
            net = load_states(net, seg_model_path)
            # net = freeze_encoder(net)
        else:
            print('Please specify the right model name')
            sys.exit(1)
        return net

    def find_object_boundary(self, mask):
        """
        Find the bounding box boundary of object in a binary mask.
        """
        rows, cols = torch.where(mask)
        min_row, max_row = torch.min(rows), torch.max(rows)
        min_col, max_col = torch.min(cols), torch.max(cols)
        return min_row.item(), max_row.item(), min_col.item(), max_col.item()
    
    def test(self, save_dir, verbose):
        # df = pd.DataFrame(columns=['id', 'dust_ratio', 'min_row', 'max_row', 'min_col', 'max_col'])
        data = {'id': [], 'dust_ratio': [], 'min_row': [], 'max_row': [], 'min_col': [], 'max_col': []}
        total_area = 512 * 1024
        for batch_idx, (batch_image_ids, inputs) in enumerate(self.testloader):
            inputs = inputs.to(self.device)
            outputs, confs = self.net(inputs)
            for image_id, pred in zip(batch_image_ids, outputs):
                # get dust pixel count and dust cloud location
                pixel_count = torch.count_nonzero(pred).item()
                if pixel_count > 0:
                    min_row, max_row, min_col, max_col = self.find_object_boundary(pred[0] == 1)
                else:
                    min_row, max_row, min_col, max_col = 0, 0, 0, 0
                # df.loc[len(df)] = [image_id, pixel_count / total_area, min_row, max_row, min_col, max_col]
                data['id'].append(image_id)
                data['dust_ratio'].append(pixel_count / total_area)
                data['min_row'].append(min_row)
                data['max_row'].append(max_row)
                data['min_col'].append(min_col)
                data['max_col'].append(max_col)
            
            if (batch_idx+1) % 100 == 0:
                logging.info(f'processed {batch_idx+1} batches.')
        
#                 # save image and output
# #                 img = np.transpose((img * 255).astype(np.uint8), (1, 2, 0))
#                 pred = (pred[0] * 255).numpy().astype(np.uint8)
# #                 canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
# #                 canvas[:512, :, :] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# #                 canvas[512:, :, 0] = pred
# #                 canvas[512:, :, 1] = pred
# #                 canvas[512:, :, 2] = pred
#                 cv2.imwrite(os.path.join(save_dir, image_id+'.png'), pred)

        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(save_dir, 'preds.csv'), index=False)

    
if __name__ == '__main__':
    color_jitter = {"brightness":0.1,"contrast":0,"saturation":0.2,"hue":0.3}
    transform_train = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
        transforms.ColorJitter(**color_jitter),
        transforms.Normalize((0.3374, 0.3408, 0.3932), (0.2072, 0.2146, 0.2453)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((512,1024)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.3374, 0.3408, 0.3932), (0.2072, 0.2146, 0.2453)),
    ])

    logging.basicConfig(level=logging.INFO)

    # model path
    model_dir = '/data/jupiter/li.yu/exps/driveable_terrain_model/'
    model_name = 'v471_rd_2cls_dustseghead_0808'
    # model_name = 'v471_rf_2cls_dustseghead_0927'
    # model_name = 'v471_debayerrgb_2cls_dustseghead_0919'
    seg_model_path = os.path.join(model_dir, model_name, 'job_quality_val_bestmodel.pth')
    save_dir = os.path.join(model_dir, model_name, 'only_new_hitchhiker_left_images_location_filtered_20221018/')
    os.makedirs(save_dir, exist_ok=True)
    
    # test data set
    data_dir = '/data/jupiter/li.yu/data/only_new_hitchhiker_left_images_location_filtered_20221018/'
    # data_dir = '/data/jupiter/datasets/2022_productivity_ts_v2_hdr/'
    test_df = pd.read_csv(os.path.join(data_dir, 'annotations.csv'), low_memory=False)
    test_df = test_df[test_df.camera_location.str.endswith('left')]
    test_df = test_df[['id', 'hdr_mode', 'artifact_debayeredrgb_0_save_path']]
    # test_df = test_df[['id', 'hdr_mode', 'artifact_debayeredrgb_0_save_path', 'stereo_pipeline_npz_save_path']]
    print(test_df.shape)
    
    testset = DustData(data_dir, test_df, transform_test, 'test', rgb='debayered', normalization_policy='percentile')
    testloader = torch.utils.data.DataLoader(testset, batch_size=48, shuffle=False, num_workers=24)

    solver = Solver(testloader, seg_model_path)
    start = time.time()
    solver.test(save_dir, verbose=True)
    end = time.time()
    print(end - start, 's')
    print()
