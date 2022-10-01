import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import csv
import pickle
import random
import argparse
import os
import numpy as np
import pandas as pd

from glob import glob
from os.path import exists, join
from tqdm import tqdm as tqdm_base

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
import sklearn.metrics
from sklearn.metrics import roc_auc_score

import utils
import torchxrayvision as xrv


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


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


def main(cfg):
    device = torch.device(cfg.device)
    
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    test_data = utils.load_data_inference(cfg)
    
    test_loader = DataLoader(test_data,
                           batch_size=cfg.batch_size,
                           shuffle=SequentialSampler(test_data),
                           num_workers=cfg.num_workers,
                           pin_memory=True,
                           drop_last=True)
    
    model = xrv.models.DenseNet(weights="all")
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    test_auc, test_loss, task_aucs = inference(name='Test',
                                     model=model,
                                     device=device,
                                     data_loader=test_loader,
                                     criterion=criterion,
                                     limit=cfg.num_batches//2
                                    )

    print(f"Average AUC for all pathologies {test_auc:4.4f}")
    print(f"Test loss: {test_loss:4.4f}")                                 
    print(f"AUC for each task {[round(x, 4) for x in task_aucs]}")


def get_args_parser():
    parser = argparse.ArgumentParser(description='X-RAY Pathology Detection')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--dataset_dir', type=str, default="./data/")
    parser.add_argument('--test_data', type=str, default="nih")
    parser.add_argument('--device', type=str, default="cpu")

    ### Data loader
    parser.add_argument('--data_resize', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--num_batches', type=int, default=430, help='')

    ### Data Augmentation                  
    parser.add_argument('--data_aug_rot', type=int, default=45, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
                        
    return parser
                        

if __name__ == "__main__":
    cfg = get_args_parser().parse_args()
    main(cfg)