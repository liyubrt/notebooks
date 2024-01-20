import os
import sys
import cv2
import random
import pickle
import shutil
import logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
logging.basicConfig(level=logging.INFO)


categorical_labels = [['objects', 50], ['humans', 50], ['vehicles', 200], ['dust', 1000], ['birds', 50], ['airborne', 50]]
columns_to_use = ['unique_id', 'artifact_debayeredrgb_0_save_path', 'annotation_pixelwise_0_save_path']

class HaloData(torch.utils.data.Dataset):
    def __init__(self, data_dir, df, transform):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # get image
        image = Image.open(os.path.join(self.data_dir, row.artifact_debayeredrgb_0_save_path))
        # label = Image.open(os.path.join(self.data_dir, row.annotation_pixelwise_0_save_path))
        image = self.transform(image)
        
        # get label
        label = np.array([row[sub] >= thres for sub,thres in categorical_labels]).astype(np.float32)

        return row.unique_id, image, label


class Solver:
    def __init__(self, dataloaders, save_dir):
        self.dataloaders = dataloaders
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = models.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, len(categorical_labels))  # reset final fully connected layer
        logging.info(f'# params : {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
        self.net = net.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()  # for multi-label classification
        self.optimizer = optim.AdamW(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=60, eta_min=1e-6)
        self.save_dir = save_dir

    def iterate(self, epoch, phase):
        self.net.train(phase == 'train')
        dataloader = self.dataloaders[phase]
        total_loss = 0
        correct = 0
        total = 0
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
        str = f'epoch {epoch}: {phase} | loss: {total_loss/(batch_idx+1):.3f} | acc: '
        for i, (sub, _) in enumerate(categorical_labels):
            str += f'{sub} {(total_acc[i] / total).item():.2f}, '
        logging.info(str)
        return total_loss, total_acc

    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            self.iterate(epoch, 'train')
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
            checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
                logging.info('best val loss found')
            logging.info('')
            model_path = os.path.join(self.save_dir, 'last.pth')
            torch.save(checkpoint, model_path)
        model_path = os.path.join(self.save_dir, 'best.pth')
        torch.save(best_checkpoint, model_path)
    
    def test_iterate(self, epoch, phase):
        self.net.train(phase == 'train')
        dataloader = self.dataloaders[phase]
        trues = []
        preds = []
        for batch_idx, (_, inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            _, outputs = outputs.max(1)
            trues.append(targets.numpy())
            preds.append(outputs.detach().cpu().numpy())            
        return np.concatenate(trues), np.concatenate(preds)

    def test(self, ori_data_dir, cat_save_dir):
        checkpoint = torch.load(self.model_path)
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        self.net.load_state_dict(checkpoint['state_dict'])
        logging.info('load model at epoch {}, with val loss: {:.3f}'.format(epoch, val_loss))
        y_true, y_pred = self.test_iterate(epoch, 'test')
        logging.info(y_true.shape, y_pred.shape)
        
        acc = accuracy_score(y_true, y_pred)
        logging.info('accuracy', acc)
        
#         self.save_pred_by_cat(y_true, y_pred, ori_data_dir, cat_save_dir)


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
    
    data_dir = '/data2/jupiter/datasets/halo_rgb_stereo_train_v6_1/'
    csv_path = os.path.join(data_dir, 'master_annotations_dedup.csv')
    label_csv_path = '/data/jupiter/li.yu/data/halo_rgb_stereo_train_test/train_v6_1_categorical_count.csv'
    
    run_id = 'v61_6cls_cat_0119'
    save_dir = f'/data/jupiter/li.yu/exps/driveable_terrain_model/{run_id}'
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f'writing checkpoints to {save_dir}')
    
    cur_df = pd.read_csv(csv_path)
    label_df = pd.read_csv(label_csv_path)
    cur_df = cur_df[columns_to_use].merge(label_df, on='unique_id')
    logging.info(f'{cur_df.shape}')
    train_df, val_df = train_test_split(cur_df, test_size=0.05, random_state=304)
    logging.info(f'{train_df.shape}, {val_df.shape}')

    trainset = HaloData(data_dir, train_df, transform_train)
    valset = HaloData(data_dir, val_df, transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=32, shuffle=False, num_workers=8)
    dataloaders = {'train':trainloader, 'val':valloader}

    # test_df = None
    # testset = HaloData(data_dir, test_df, transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=40, shuffle=False, num_workers=8)
    # dataloaders = {'test':testloader}
    
    solver = Solver(dataloaders, save_dir)
    solver.train(60)
    # solver.test(ori_data_dir, cat_save_dir)
    logging.info('')
