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

import torchxrayvision.models as models
import torchxrayvision.datasets as datasets
from densenet import densenet100, densenet121 #https://github.com/prigoyal/pytorch_memonger/blob/master/models/optimized/densenet_new.py
from agc import adaptive_clip_grad      #https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
from pytorchtools import EarlyStopping  #https://github.com/Bjarten/early-stopping-pytorch
from madgrad import MADGRAD             #https://github.com/facebookresearch/madgrad

parser = argparse.ArgumentParser(description='X-RAY Pathology Detection - Using Risk Extrapolation')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--name', type=str, default="valid")
parser.add_argument('--output_dir', type=str, default="rex_chest_xray/")
parser.add_argument('--model_name', type=str, default="densenet121")
parser.add_argument('--dataset_name', type=str, default="nih_mc_cx")
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--num_epochs', type=int, default=250, help='')  
parser.add_argument('--patience', type=int, default=10, help='')

### Data loader
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--shuffle', type=bool, default=False, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--num_batches', type=int, default=500, help='')

### Data Augmentation                  
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')

### REx hyperparameters
parser.add_argument('--erm_amount', type=float, default=1.0)
parser.add_argument('--l2_regularizer_weight', type=float, default=1e-5) 
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=10000.0)

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

############################################# Data Loading and Processing #################################################
### Data Augmentations
data_aug = torchvision.transforms.Compose([
        datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
                                            
        torchvision.transforms.ToTensor()
        ])
print(data_aug)

transforms = torchvision.transforms.Compose([datasets.XRayCenterCrop(), datasets.XRayResizer(224)])


train_datas = []
datas_names = []

### Load NIH Dataset ### 
NIH_dataset = datasets.NIH_Dataset(
        imgpath="images-224-NIH", 
        csvpath="Data_Entry_2017_v2020.csv.gz",
        bbox_list_path="BBox_List_2017.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False)
train_datas.append(NIH_dataset)
datas_names.append("nih")

# ### Load MIMIC_CH Dataset ###
MIMIC_CH_dataset = datasets.MIMIC_Dataset(
    imgpath="images-224-MIMIC/files",
    csvpath="MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
    metacsvpath="MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
    transform=transforms, data_aug=data_aug, unique_patients=False)
train_datas.append(MIMIC_CH_dataset)
datas_names.append("mimic_ch")

## Load CHEXPERT Dataset ###
CHEX_dataset = datasets.CheX_Dataset(
        imgpath="CheXpert-v1.0-small",
        csvpath="CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)
train_datas.append(CHEX_dataset)
datas_names.append("chex")

### Load PADCHEST Dataset ###
PC_dataset = datasets.PC_Dataset(
        imgpath="PC/images-224",
        csvpath="PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)

### Load OPENI Dataset ###
OPENI_dataset = datasets.Openi_Dataset(
        imgpath="images-openi/",
        xmlpath="NLMCXR_reports.tgz", 
        dicomcsv_path="nlmcxr_dicom_metadata.csv.gz",
        tsnepacsv_path="nlmcxr_tsne_pa.csv.gz",
        transform=transforms, data_aug=None)

datasets.default_pathologies = ['Effusion', 'Cardiomegaly', 'Edema']
print(f"Common pathologies among all train and validation datasets: {datasets.default_pathologies}")

for d in train_datas:
    datasets.relabel_dataset(datasets.default_pathologies, d)

datasets.relabel_dataset(datasets.default_pathologies, PC_dataset)
datasets.relabel_dataset(datasets.default_pathologies, OPENI_dataset)

np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

    
def worker_init_fn(worker_id):
    np.random.seed(cfg.seed + worker_id)

def data_loaders(cfg, num_batches, train_datas, valid_data, test_data=None):  
    train_loaders = [[{} for i in range(num_batches)] for i in range(len(train_datas))]

    for train_loader in train_loaders:
        for data in train_datas:
            if train_loaders.index(train_loader) == train_datas.index(data):
                tr_l = DataLoader(data,
                               batch_size=cfg.batch_size,
                               shuffle=True,
                               num_workers=cfg.num_workers, 
                               worker_init_fn=worker_init_fn,
                               pin_memory=True,
                               drop_last=True)
                train_loader.insert(0, tr_l)

    valid_loader = DataLoader(valid_data,
                           batch_size=cfg.batch_size,
                           shuffle=False,
                           num_workers=cfg.num_workers,
                           worker_init_fn=worker_init_fn,
                           pin_memory=True,
                           drop_last=True)

    test_loader = DataLoader(test_data,
                           batch_size=cfg.batch_size,
                           shuffle=False,
                           num_workers=cfg.num_workers,
                           worker_init_fn=worker_init_fn, 
                           pin_memory=True,
                           drop_last=True)    

    return train_loaders, valid_loader, test_loader

train_loaders, valid_loader, test_loader = data_loaders(cfg=cfg, num_batches=cfg.num_batches, 
                                           train_datas=train_datas, valid_data=PC_dataset, test_data=OPENI_dataset)


################################### Model, Optimization and Criterion #####################################################
model = densenet121(num_classes=len(datasets.default_pathologies))
model = model.to(device)

optimizer = MADGRAD( model.parameters(), lr=cfg.lr)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, epochs=cfg.num_epochs, steps_per_epoch=cfg.num_batches)        

################################################### Training and Validation #############################################
def compute_loss(outputs, targets, train_loader, device):    
    weights = np.nansum(train_loader.dataset.labels, axis=0)
    weights = weights.max() - weights + weights.mean()
    weights = weights/weights.max()                    
    weights = torch.from_numpy(weights).to(device).float()
    
    loss = torch.zeros(1).to(device).float()
    for task in range(targets.shape[1]):
        task_output = outputs[:,task]
        task_target = targets[:,task]
        mask = ~torch.isnan(task_target)
        task_output = task_output[mask]
        task_target = task_target[mask]
        if len(task_target) > 0:
            task_loss = criterion(task_output.float(), task_target.float())
            loss += weights[task]*task_loss
            
    return loss.sum()

def train_epoch(num_batches, model, train_loaders, epoch, device, optimizer):
    model.train()
    avg_loss = []
    t = tqdm(range(1, num_batches+1))
    for step in t:
        optimizer.zero_grad(set_to_none=True)
        for idx, train_loader in enumerate(train_loaders):
            dataloader_iterator = iter(train_loader[0])
            sample = next(dataloader_iterator)
            image, target = sample["img"].float().to(device), sample["lab"].to(device)
            
            outputs = model(image)

            train_loader[step]["loss"] = compute_loss(outputs, target, train_loader[0], device)

        train_nll = torch.stack([train_loaders[0][step]['loss'].detach(), train_loaders[1][step]['loss'].detach(), train_loaders[2][step]['loss'].detach()]).mean()

        if cfg.cuda:
            weight_norm = torch.as_tensor(0., device=device)
        else:
            weight_norm = torch.as_tensor(0.)

        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss1 = train_loaders[0][step]['loss']
        loss2 = train_loaders[1][step]['loss']
        loss3 = train_loaders[2][step]['loss']

        ### early loss mean
        loss1 = loss1.mean()
        loss2 = loss2.mean()
        loss3 = loss3.mean()

        loss = 0.0
        loss += cfg.erm_amount * (loss1 + loss2 + loss3)

        loss += cfg.l2_regularizer_weight * weight_norm

        penalty_weight = (cfg.penalty_weight if step >= cfg.penalty_anneal_iters else 1.0)

        rex_penalty = torch.std(torch.as_tensor([[loss1, loss2, loss3]]), unbiased=False)

        #rex
        loss += penalty_weight * rex_penalty

        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        loss.backward()

        avg_loss.append(train_nll.detach())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {torch.as_tensor(avg_loss).mean():4.4f}')
        
        adaptive_clip_grad(model.parameters(), clip_factor=0.01, eps=1e-3, norm_type=2.0)
        optimizer.step()
        scheduler.step()
    
    return torch.as_tensor(avg_loss).mean()


def valid_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
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
                
                task_outputs[task].append(task_output.detach())
                task_targets[task].append(task_target.detach())

            loss = loss.sum()
            
            avg_loss.append(loss.detach())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {torch.as_tensor(avg_loss).mean():4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return torch.as_tensor(avg_loss).mean(), auc


############################################# Training #################################################
def rex_train(model, num_epochs, patience):        
    dataset_name = cfg.dataset_name + "-" + cfg.model_name + "-" + cfg.name

    print(f'Using device: {device}')
    print(f"Output directory: {cfg.output_dir}")

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(cfg.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    train_losses = [] 
    valid_losses = []
    valid_aucs = []
    
    for epoch in range(start_epoch, num_epochs):
        np.random.seed(cfg.seed + epoch)
        train_loss = train_epoch(num_batches=cfg.num_batches,
                               model=model,
                               train_loaders=train_loaders,
                               epoch=epoch,
                               device=device,
                               optimizer=optimizer)
        
        valid_loss, valid_auc = valid_epoch(name='Valid',
                                                 epoch=epoch,
                                                 model=model,
                                                 device=device,
                                                 data_loader=valid_loader,
                                                 criterion=criterion,
                                                 limit=cfg.num_batches)


        train_losses.append(train_loss) 
        valid_losses.append(valid_loss)
        valid_aucs.append(valid_auc)
                               
        if np.mean(valid_auc) > best_metric:
            best_metric = np.mean(valid_auc)
            weights_for_best_validauc = model.state_dict()
            optimizer_for_best_validauc = optimizer.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best.pt'))

        stat = {
            "epoch": epoch + 1,
            "trainloss": train_loss,
            "validauc": valid_auc,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        torch.save(model, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))
        
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, valid_losses, valid_aucs

train_losses, valid_losses, valid_aucs = rex_train(model=model, num_epochs=cfg.num_epochs, patience=cfg.patience)

with open('rex_trainVal_results.csv', 'w') as csvfile:
    field_names = ['Train_loss', 'Valid_loss', 'Valid_AUC']
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    
    writer.writeheader()
    
    for train_loss, val_loss, val_auc in zip(train_losses, valid_losses, valid_aucs):
        writer.writerow({'Train_loss': train_loss, 
                     'Valid_loss': val_loss, 
                     'Valid_AUC': val_auc,
                     })

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
                
                task_outputs[task].append(task_output.detach())
                task_targets[task].append(task_target.detach())

            loss = loss.sum()
            
            avg_loss.append(loss.detach())
            
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

    return auc, torch.Tensor(avg_loss).mean(), task_aucs

model = densenet121(num_classes=len(datasets.default_pathologies))
model.load_state_dict(torch.load("./rex_chest_xray/nih_mc_cx-densenet121-valid-best.pt").state_dict())

test_auc, test_loss, task_aucs = inference(name='Test',
                                         model=model,
                                         device=device,
                                         data_loader=test_loader,
                                         criterion=criterion)

print(f"Average AUC for all pathologies {test_auc}")
print(f"Test loss: {test_loss}")                                 
print(f"AUC for each task {task_aucs}")
                                 
with open('rex_test_results.csv', 'w') as csvfile:
    field_names = ['Test_loss', 'Test_AVG_AUC', 'Task_AUCs']
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    
    writer.writeheader()
    writer.writerow({'Test_loss': test_loss, 
                     'Test_AVG_AUC': test_auc, 
                     'Task_AUCs': task_aucs})
