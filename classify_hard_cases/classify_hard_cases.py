import os
import sys
import cv2
import time
import random
import shutil
import logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
logging.basicConfig(level=logging.INFO)


# Use binary label or percentage label
BINARY_LABEL = True

# 6-class
if BINARY_LABEL:
    categorical_labels = [['objects', 50], ['humans', 50], ['vehicles', 200], ['dust', 1000], ['birds', 25], ['airborne', 50]]
else:
    categorical_labels = [['objects', 24664], ['humans', 31081], ['vehicles', 131716], ['dust', 360000], ['birds', 2256], ['airborne', 5477]]
# pos_weight = [1.0, 5.0, 6.0, 7.0, 180.0, 15.0]

# # 2-class
# if BINARY_LABEL:
#     categorical_labels = [['birds', 25], ['airborne', 50]]
# else:
#     categorical_labels = [['birds', 2256], ['airborne', 5477]]
# pos_weight = [12.0, 1.0]


class HaloData(torch.utils.data.Dataset):
    def __init__(self, data_dir, df, transform, phase):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # get image
        if phase == 'train':
            image = Image.open(row.artifact_debayeredrgb_0_save_path)
        else:
            try:
                image = Image.open(os.path.join(self.data_dir, row.artifact_debayeredrgb_0_save_path))
            except:
                image = Image.new('RGB', (1944, 1204))
            # label = Image.open(os.path.join(self.data_dir, row.annotation_pixelwise_0_save_path))
        image = self.transform(image)
        
        # get label
        if phase == 'train':
            if BINARY_LABEL:
                label = np.array([row[sub] >= thres for sub,thres in categorical_labels]).astype(np.float32)
            else:
                label = np.clip(np.array([row[sub] / thres for sub,thres in categorical_labels]), None, 1.0).astype(np.float32)
            return row.unique_id, image, label
        else:
            # return row.unique_id, image
            return row.id, image


class Solver:
    def __init__(self, dataloaders, model_dir, pos_weight):
        self.dataloaders = dataloaders
        self.epochs = 60
        self.early_stop = 7
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = models.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, len(categorical_labels))  # reset final fully connected layer
        logging.info(f'use {torch.cuda.device_count()} gpus')
        logging.info(f'# params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
        self.net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(self.device)
        if BINARY_LABEL:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        else:
            self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.model_dir = model_dir

    def iterate(self, epoch, phase):
        self.net.train(phase == 'train')
        dataloader = self.dataloaders[phase]
        total_loss = 0
        correct = 0
        total = 0
        t1 = time.time()
        for batch_idx, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            total += targets.size(0)
            if BINARY_LABEL:
                outputs = torch.sigmoid(outputs)
                correct += (outputs > 0.5).eq(targets).sum(dim=0)  # C, binary label
            else:
                correct += (outputs > 0.01).eq((targets > 0.01)).sum(dim=0)  # C, percentage label
        if phase == 'train':
            self.scheduler.step()
        total_acc = correct / total
        t2 = time.time()
        str = f'epoch {epoch}: {phase} | {int((t2-t1)/60)}m | loss: {total_loss/(batch_idx+1):.3f} | acc: '
        for i, (sub, _) in enumerate(categorical_labels):
            str += f'{sub} {total_acc[i].item():.2f}, '
        logging.info(str)
        return total_loss, total_acc

    def train(self):
        best_loss = float('inf')
        last_best = 0
        for epoch in range(self.epochs):
            self.iterate(epoch, 'train')
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
            checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
            if val_loss < best_loss:
                best_loss = val_loss
                last_best = epoch
                best_checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
                logging.info('best val loss found')
            torch.save(checkpoint, os.path.join(self.model_dir, 'last.pth'))
            if epoch - last_best >= self.early_stop:
                logging.info('early stop')
                break
            logging.info('')
        torch.save(best_checkpoint, os.path.join(self.model_dir, 'best.pth'))
    
    def test_iterate(self, phase):
        self.net.train(phase == 'train')
        dataloader = self.dataloaders[phase]
        ids, preds = [], []
        for batch_idx, (unique_ids, inputs) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            outputs = torch.sigmoid(outputs)
            preds.append(outputs.detach().cpu().numpy())   
            ids.append(unique_ids)
            if (batch_idx + 1) % 10 == 0:
                logging.info(f'processed {(batch_idx + 1) * outputs.size(0)} images')
        return np.concatenate(ids), np.concatenate(preds)

    def test(self, save_dir):
        checkpoint = torch.load(os.path.join(self.model_dir, 'best.pth'), map_location=lambda storage, loc: storage)
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        self.net.load_state_dict(checkpoint['state_dict'])
        logging.info('load model at epoch {}, with val loss: {:.3f}'.format(epoch, val_loss))
        ids, y_pred = self.test_iterate('test')
        logging.info(f'pred shape {y_pred.shape}')
        results = {'unique_id': ids}
        results.update({categorical_labels[i][0]: y_pred[:,i] for i in range(len(categorical_labels))})
        df = pd.DataFrame(data=results)
        df.to_csv(os.path.join(save_dir, 'output.csv'), index=False)


def load_dataset(data_dir, csv_path, label_csv_path, scale_factor=1):
    columns_to_use = ['unique_id', 'artifact_debayeredrgb_0_save_path', 'annotation_pixelwise_0_save_path']
    cur_df = pd.read_csv(csv_path)
    label_df = pd.read_csv(label_csv_path)
    cur_df = cur_df[columns_to_use].merge(label_df, on='unique_id')
    for i in range(1, 3):
        cur_df[columns_to_use[i]] = cur_df[columns_to_use[i]].apply(lambda s: os.path.join(data_dir, s))
    logging.info(f'{cur_df.shape} from {csv_path}')
    return cur_df

def get_pos_weight(df, categorical_labels):
    pos_weight = np.array([len(df[df[cat] > thres]) for cat, thres in categorical_labels])
    pos_weight = pos_weight.max() / pos_weight
    return pos_weight


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.Resize((512,1024)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.2,hue=0.3),
        transforms.ToTensor(),
        # transforms.Normalize((0.3374, 0.3408, 0.3932), (0.2072, 0.2146, 0.2453)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((512,1024)),
        transforms.ToTensor(),
        # transforms.Normalize((0.3374, 0.3408, 0.3932), (0.2072, 0.2146, 0.2453)),
    ])

    if len(sys.argv) < 3:
        logging.info('Please input the run_id and train/test phase to start')
        sys.exit()
    run_id = sys.argv[1]
    phase = sys.argv[2]
    model_dir = f'/data/jupiter/li.yu/exps/driveable_terrain_model/{run_id}'

    if phase == 'train':
        # os.makedirs(model_dir, exist_ok=True)
        logging.info(f'writing checkpoints to {model_dir}')

        # load halo data
        data_dir = '/data2/jupiter/datasets/halo_rgb_stereo_train_v6_1/'
        csv_path = os.path.join(data_dir, 'master_annotations_dedup.csv')
        label_csv_path = '/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v6_1_categorical_count.csv'
        cur_df = load_dataset(data_dir, csv_path, label_csv_path)

        # load rev1 data
        rev1_data_dir = '/data/jupiter/li.yu/data/Jupiter_train_v6_2_birds_airborne_debris'
        rev1_csv_path = os.path.join(rev1_data_dir, 'master_annotations.csv')
        rev1_label_csv_path = '/data/jupiter/li.yu/data/Jupiter_train_v6_2/train_v6_2_birds_airborne_categorical_count.csv'
        rev1_cur_df = load_dataset(rev1_data_dir, rev1_csv_path, rev1_label_csv_path)
        
        # merge data
        cur_df = pd.concat([cur_df, rev1_cur_df], ignore_index=True)
        logging.info(f'{cur_df.shape} after merging two datasets')

        # get pos_weight
        pos_weight = get_pos_weight(cur_df, categorical_labels)
        logging.info(f'pos_weight: {list(pos_weight)}')

        # train val split
        train_df, val_df = train_test_split(cur_df, test_size=0.05, random_state=304)
        logging.info(f'{train_df.shape}, {val_df.shape}')

        train_set = HaloData(None, train_df, transform_train, phase)
        val_set = HaloData(None, val_df, transform_test, phase)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True, num_workers=16)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=64, shuffle=False, num_workers=16)
        dataloaders = {'train':train_loader, 'val':val_loader}
        
        solver = Solver(dataloaders, model_dir, torch.tensor(pos_weight))
        solver.train()
    else:
        # dataset = '20240119_halo_rgb_stereo'
        # dataset = '20231219_halo_rgb_stereo'
        # dataset = 'halo_rgb_stereo_test_v6_1'
        dataset = 'halo_potential_airborne_debris_from_train_6_2'
        data_dir = f'/data2/jupiter/datasets/{dataset}/'
        # csv_path = os.path.join(data_dir, 'master_annotations_dedup.csv')
        # csv_path = os.path.join(data_dir, 'annotations_left.csv')
        csv_path = os.path.join(data_dir, 'annotations.csv')
        save_dir = f'/data/jupiter/li.yu/exps/driveable_terrain_model/{run_id}/{dataset}'
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f'writing results to {save_dir}')
        test_df = pd.read_csv(csv_path)
        logging.info(f'{test_df.shape}')
        test_set = HaloData(data_dir, test_df, transform_test, phase)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=16)
        dataloaders = {'test':test_loader}
    
        solver = Solver(dataloaders, model_dir, None)
        solver.test(save_dir)
    
    logging.info('done.')
