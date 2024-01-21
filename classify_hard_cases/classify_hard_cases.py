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


# categorical_labels = [['objects', 50], ['humans', 50], ['vehicles', 200], ['dust', 1000], ['birds', 50], ['airborne', 50]]
categorical_labels = [['objects', 24664], ['humans', 31081], ['vehicles', 131716], ['dust', 360000], ['birds', 2256], ['airborne', 5477]]
columns_to_use = ['unique_id', 'artifact_debayeredrgb_0_save_path', 'annotation_pixelwise_0_save_path']

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
        image = Image.open(os.path.join(self.data_dir, row.artifact_debayeredrgb_0_save_path))
        # label = Image.open(os.path.join(self.data_dir, row.annotation_pixelwise_0_save_path))
        image = self.transform(image)
        
        # get label
        if phase == 'train':
            # label = np.array([row[sub] >= thres for sub,thres in categorical_labels]).astype(np.float32)
            label = np.clip(np.array([row[sub] / thres for sub,thres in categorical_labels]), None, 1.0).astype(np.float32)
            return row.unique_id, image, label
        else:
            return row.unique_id, image


class Solver:
    def __init__(self, dataloaders, model_dir):
        self.dataloaders = dataloaders
        self.epochs = 60
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = models.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, len(categorical_labels))  # reset final fully connected layer
        logging.info(f'# params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
        self.net = torch.nn.DataParallel(net, device_ids=[0,1,2,3]).to(self.device)
        pos_weight = torch.tensor([1.0, 5.0, 6.0, 7.0, 180.0, 15.0])  # set to None to disable
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)  # for multi-label classification
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
            outputs = outputs > 0.0  # outputs = torch.sigmoid(outputs) and outputs > 0.5
            total += targets.size(0)
            correct += outputs.eq(targets).sum(dim=0)  # C
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
        for epoch in range(self.epochs):
            self.iterate(epoch, 'train')
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
            checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
                logging.info('best val loss found')
            logging.info('')
            torch.save(checkpoint, os.path.join(self.model_dir, 'last.pth'))
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
        checkpoint = torch.load(os.path.join(self.model_dir, 'last.pth'), map_location=lambda storage, loc: storage)
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


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.Resize((512,1024)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.2,hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.3374, 0.3408, 0.3932), (0.2072, 0.2146, 0.2453)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((512,1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.3374, 0.3408, 0.3932), (0.2072, 0.2146, 0.2453)),
    ])

    if len(sys.argv) < 3:
        logging.info('Please input the run_id and train/test phase to start')
        sys.exit()
    run_id = sys.argv[1]
    phase = sys.argv[2]
    model_dir = f'/data/jupiter/li.yu/exps/driveable_terrain_model/{run_id}'

    if phase == 'train':
        data_dir = '/data2/jupiter/datasets/halo_rgb_stereo_train_v6_1/'
        csv_path = os.path.join(data_dir, 'master_annotations_dedup.csv')
        label_csv_path = '/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v6_1_categorical_count.csv'
        
        # os.makedirs(model_dir, exist_ok=True)
        logging.info(f'writing checkpoints to {model_dir}')
        
        cur_df = pd.read_csv(csv_path)
        label_df = pd.read_csv(label_csv_path)
        cur_df = cur_df[columns_to_use].merge(label_df, on='unique_id')
        logging.info(f'{cur_df.shape}')
        train_df, val_df = train_test_split(cur_df, test_size=0.05, random_state=304)
        logging.info(f'{train_df.shape}, {val_df.shape}')

        train_set = HaloData(data_dir, train_df, transform_train, phase)
        val_set = HaloData(data_dir, val_df, transform_test, phase)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True, num_workers=16)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=64, shuffle=False, num_workers=16)
        dataloaders = {'train':train_loader, 'val':val_loader}
        
        solver = Solver(dataloaders, model_dir)
        solver.train()
    else:
        data_dir = '/data2/jupiter/datasets/halo_rgb_stereo_test_v6_1/'
        csv_path = os.path.join(data_dir, 'master_annotations_dedup.csv')
        save_dir = f'/data/jupiter/li.yu/exps/driveable_terrain_model/{run_id}/halo_rgb_stereo_test_v6_1'
        os.makedirs(save_dir, exist_ok=True)
        test_df = pd.read_csv(csv_path)
        logging.info(f'{test_df.shape}')
        test_set = HaloData(data_dir, test_df, transform_test, phase)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=24, shuffle=False, num_workers=8)
        dataloaders = {'test':test_loader}
    
        solver = Solver(dataloaders, model_dir)
        solver.test(save_dir)
    
    logging.info('done.')
