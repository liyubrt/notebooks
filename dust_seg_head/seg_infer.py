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
from utils import normalize_image, get_tire_mask


class BRTData(torch.utils.data.Dataset):
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
        camera_location = sample_df.camera_location
        
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
        return image_id, image, camera_location


class Solver:
    def __init__(self, testloader, seg_model_path, side_left_mask_path, side_right_mask_path):
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.load_model('brt', seg_model_path)  # resnet18 or brt
        print('# tunable params', sum(p.numel() for p in net.parameters() if p.requires_grad))
        net = nn.DataParallel(net)
        self.net = net.to(self.device)
        self.net.eval()
        self.side_left_mask = get_tire_mask(side_left_mask_path)
        self.side_right_mask = get_tire_mask(side_right_mask_path)

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
            params = {"input_dims": 3, "num_classes": 7, "seg_output": True, "cls_output": False, "add_softmax_layer": True,
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
    
    def apply_tire_mask(self, label: np.ndarray, camera_location: AnyStr) -> np.ndarray:
        """
        This function applies the appropriate mask image to a given label array depending on the input camera location
        """
        if camera_location == 'side-left-left':
            label[self.side_left_mask] = 255
        elif camera_location == 'side-right-left':
            label[self.side_right_mask] = 255
        return label

    def test(self, save_dir):
        data = {'id': [], 
                'lo_pixel_count': [], 'lo_min_row': [], 'lo_max_row': [], 'lo_min_col': [], 'lo_max_col': [],
                'trees_pixel_count': [], 'trees_min_row': [], 'trees_max_row': [], 'trees_min_col': [], 'trees_max_col': [],
                'human_pixel_count': [], 'human_min_row': [], 'human_max_row': [], 'human_min_col': [], 'human_max_col': [],
                'vehicle_pixel_count': [], 'vehicle_min_row': [], 'vehicle_max_row': [], 'vehicle_min_col': [], 'vehicle_max_col': [],
                'mergeconf_pixel_count': [], 'mergeconf_min_row': [], 'mergeconf_max_row': [], 'mergeconf_min_col': [], 'mergeconf_max_col': [],
                }
        class_kw_map = {1: 'lo', 3: 'trees', 5: 'human', 6: 'vehicle', -1: 'mergeconf'}
        mergeconf_thres = 0.35
        for batch_idx, (image_ids, inputs, camera_locations) in enumerate(self.testloader):
            inputs = inputs.to(self.device)
            outputs, softmax_logits = self.net(inputs)
            mergeconf_softmaxs = softmax_logits[:, [c for c in class_kw_map.keys() if c != -1]].sum(dim=1)
            for image_id, camera_location, pred, mergeconf_softmax in zip(image_ids, camera_locations, outputs, mergeconf_softmaxs):
                data['id'].append(image_id)
                pred = self.apply_tire_mask(pred, camera_location)
                # get pixel count and location
                for class_id, kw in class_kw_map.items():
                    if class_id != -1:
                        pred_mask = pred == class_id
                    else:
                        pred_mask = mergeconf_softmax > mergeconf_thres
                    pixel_count = torch.count_nonzero(pred_mask).item()
                    if pixel_count > 0:
                        min_row, max_row, min_col, max_col = self.find_object_boundary(pred_mask)
                    else:
                        min_row, max_row, min_col, max_col = 0, 0, 0, 0
                    data[f'{kw}_pixel_count'].append(pixel_count)
                    data[f'{kw}_min_row'].append(min_row)
                    data[f'{kw}_max_row'].append(max_row)
                    data[f'{kw}_min_col'].append(min_col)
                    data[f'{kw}_max_col'].append(max_col)
            
                # # save prediction
                # save_path = os.path.join(save_dir, 'preds', image_id+'.png')
                # cv2.imwrite(save_path, pred.cpu().numpy().astype(np.uint8))

            if (batch_idx+1) % 100 == 0:
                logging.info(f'processed {batch_idx+1} batches.')

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

    side_left_mask_path = '/home/li.yu/code/JupiterEmbedded/src/dnn_engine/data/side_left_mask.png'
    side_right_mask_path = '/home/li.yu/code/JupiterEmbedded/src/dnn_engine/data/side_right_mask.png'

    # model path
    model_dir = '/data/jupiter/li.yu/exps/driveable_terrain_model/'
    model_name = 'v492_7cls_humancnp_rgb_1212'
    seg_model_path = os.path.join(model_dir, model_name, 'vehicle_cls_val_bestmodel.pth')
    save_dir = os.path.join(model_dir, model_name, 'Jupiter_March2022_VT_0_1_pass/')
    os.makedirs(save_dir, exist_ok=True)
    
    # test data set
    data_dir = '/data/jupiter/li.yu/data/Jupiter_March2022_VT_0_1_pass/'
    # data_dir = '/data/jupiter/datasets/2022_productivity_ts_v2_hdr/'
    test_df = pd.read_csv(os.path.join(data_dir, 'annotations.csv'), low_memory=False)
    test_df = test_df[test_df.camera_location.str.endswith('left')]
    test_df = test_df[['id', 'hdr_mode', 'artifact_debayeredrgb_0_save_path', 'camera_location']]
    # test_df = test_df[['id', 'hdr_mode', 'artifact_debayeredrgb_0_save_path', 'stereo_pipeline_npz_save_path']]
    print(test_df.shape)
    
    testset = BRTData(data_dir, test_df, transform_test, 'test', rgb='debayered', normalization_policy='percentile')
    testloader = torch.utils.data.DataLoader(testset, batch_size=48, shuffle=False, num_workers=24)

    solver = Solver(testloader, seg_model_path, side_left_mask_path, side_right_mask_path)
    start = time.time()
    solver.test(save_dir)
    end = time.time()
    print(end - start, 's')
    print()
