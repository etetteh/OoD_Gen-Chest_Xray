import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import csv
import pickle
import PIL
import pprint
import random
import argparse
import os,sys,inspect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from os.path import exists, join
from tqdm import tqdm as tqdm_base

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms
import skimage.transform
import sklearn.metrics
import sklearn, sklearn.model_selection
from sklearn.metrics import roc_auc_score, accuracy_score

import torchxrayvision as xrv


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='X-RAY Pathology Detection')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--dataset_dir', type=str, default="./data/")
parser.add_argument('--dataset_name', type=str, default="nih")

### Data loader
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=False, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--num_batches', type=int, default=430, help='')

### Data Augmentation                  
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')


cfg = parser.parse_args()
print(cfg)

def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

device = 'cuda' if cfg.cuda else 'cpu'
if not torch.cuda.is_available() and cfg.cuda:
    device = 'cpu'
    print("WARNING: cuda was requested but is not available, using cpu instead.")
print(f'Using device: {device}')


transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(112)])

if "nih" in cfg.dataset_name:
    ### Load NIH Dataset ### 
    NIH_dataset = xrv.datasets.NIH_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-NIH", 
            csvpath=cfg.dataset_dir + "/Data_Entry_2017_v2020.csv.gz",
            bbox_list_path=cfg.dataset_dir + "/BBox_List_2017.csv.gz",
            transform=transforms, data_aug=None, unique_patients=False)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, NIH_dataset)
    test_data = NIH_dataset

if "mc" in cfg.dataset_name:
    # ### Load MIMIC_CH Dataset ###
    MIMIC_CH_dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=cfg.dataset_dir + "/images-224-MIMIC/files",
        csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=None, unique_patients=False)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, MIMIC_CH_dataset)
    test_data = MIMIC_CH_dataset 

if "cx" in cfg.dataset_name:
    ## Load CHEXPERT Dataset ###
    CHEX_dataset = xrv.datasets.CheX_Dataset(
            imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
            csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
            transform=transforms, data_aug=None, unique_patients=False)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, CHEX_dataset)
    test_data = CHEX_dataset

if "pc" in cfg.dataset_name:
    ### Load PADCHEST Dataset ###
    PC_dataset = xrv.datasets.PC_Dataset(
            imgpath=cfg.dataset_dir + "/PC/images-224",
            csvpath=cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            transform=transforms, data_aug=None, unique_patients=False)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, PC_dataset)
    test_data = PC_dataset

if "gg" in cfg.dataset_name:
    ### Load GOOGLE Dataset ###
    GOOGLE_dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-NIH",
            csvpath=cfg.dataset_dir + "/google2019_nih-chest-xray-labels.csv.gz",
            transform=transforms, data_aug=None
            )
    xrv.datasets.default_pathologies = ['Pneumothorax', 'Fracture']
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, GOOGLE_dataset)
    test_data = GOOGLE_dataset

if "op" in cfg.dataset_name:
    ### Load OPENI Dataset ###
    OPENI_dataset = xrv.datasets.Openi_Dataset(
            imgpath=cfg.dataset_dir + "/images-openi/",
            xmlpath=cfg.dataset_dir + "/NLMCXR_reports.tgz", 
            dicomcsv_path=cfg.dataset_dir + "/nlmcxr_dicom_metadata.csv.gz",
            tsnepacsv_path=cfg.dataset_dir + "/nlmcxr_tsne_pa.csv.gz",
            transform=transforms, data_aug=None
            )
    xrv.datasets.default_pathologies = ['Effusion', 'Cardiomegaly', 'Edema']
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, OPENI_dataset)
    test_data = OPENI_dataset

if "rs" in cfg.dataset_name:    
    ### Load RSNA Dataset ###
    RSNA_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        csvpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_labels.csv",
        dicomcsvpath=cfg.dataset_dir + "/kaggle_stage_2_train_images_dicom_headers.csv.gz",
        transform=transforms, data_aug=None, unique_patients=False
        )
    xrv.datasets.default_pathologies = ['Lung Opacity', 'Pneumonia']
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, RSNA_dataset)
    test_data = RSNA_dataset

print(f"Common pathologies among all train and validation datasets: {xrv.datasets.default_pathologies}")
    
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
test_loader = DataLoader(test_data,
                       batch_size=cfg.batch_size,
                       shuffle=False,
                       num_workers=cfg.num_workers,
                       pin_memory=True,
                       drop_last=True)

###################################### Test ######################################
def inference(name, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx >= limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'{name} - Avg AUC = {auc:4.4f}')

    return auc, np.mean(avg_loss), task_aucs

model = xrv.models.DenseNet(weights="all")
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()

test_auc, test_loss, task_aucs = inference(name='Test',
                                         model=model,
                                         device=device,
                                         data_loader=test_loader,
                                         criterion=criterion,
                                         limit=cfg.num_batches//2)

print(f"Average AUC for all pathologies {test_auc:4.4f}")
print(f"Test loss: {test_loss:4.4f}")                                 
print(f"AUC for each task {[round(x, 4) for x in task_aucs]}")

