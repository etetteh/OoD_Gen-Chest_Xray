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
parser.add_argument('--seed', type=int, default=99, help='')
parser.add_argument('--model_name', type=str, default="densenet121")
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--num_epochs', type=int, default=200, help='')

parser.add_argument('--split', type=int, default=0, help='')
parser.add_argument('--valid_data', type=str, default="nih", help='')
parser.add_argument('--baseline', type=str2bool, default=True)

### Data loader
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=False, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--num_batches', type=int, default=420, help='')

### Data Augmentation                  
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')

### REx hyperparameters
parser.add_argument('--erm_amount', type=float, default=1.0)
parser.add_argument('--l2_regularizer_weight', type=float, default=1e-5) 
parser.add_argument('--penalty_weight', type=float, default=100.0)

cfg = parser.parse_args()
print(cfg)

penalty_anneal_iters = cfg.num_batches//5

if cfg.baseline:
    print("\n Training Baseline Model \n")
    output_dir = "baseline_small_split-" + str(cfg.split) + "_valid-" + cfg.valid_data + "/"
else:
    print("\n Training REx Model \n")
    output_dir = "rex_small_split-" + str(cfg.split) + "_valid-" + cfg.valid_data + "_pen-" + str(int(cfg.penalty_weight)) + "/"

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
    
### Data Augmentation    
data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
                                            
        torchvision.transforms.ToTensor()
        ])
print(data_aug)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(112)])


################################ Data Loading and Environment creation ####################### 
### Load NIH Dataset ### 
NIH_dataset = xrv.datasets.NIH_Dataset(
        imgpath="data/images-224-NIH", 
        csvpath="data/Data_Entry_2017_v2020.csv.gz",
        bbox_list_path="data/BBox_List_2017.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False)

# ### Load MIMIC_CH Dataset ###
MIMIC_CH_dataset = xrv.datasets.MIMIC_Dataset(
    imgpath="data/images-224-MIMIC/files",
    csvpath="data/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
    metacsvpath="data/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
    transform=transforms, data_aug=data_aug, unique_patients=False)

## Load CHEXPERT Dataset ###
CHEX_dataset = xrv.datasets.CheX_Dataset(
        imgpath="data/CheXpert-v1.0-small",
        csvpath="data/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)

### Load PADCHEST Dataset ###
PC_dataset = xrv.datasets.PC_Dataset(
        imgpath="data/PC/images-224",
        csvpath="data/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)

### Load GOOGLE Dataset ###
GOOGLE_dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath="data/images-224-NIH",
        csvpath="data/google2019_nih-chest-xray-labels.csv.gz",
        transform=transforms, data_aug=data_aug)

### Load OPENI Dataset ###
OPENI_dataset = xrv.datasets.Openi_Dataset(
        imgpath="data/images-openi/",
        xmlpath="data/NLMCXR_reports.tgz", 
        dicomcsv_path="data/nlmcxr_dicom_metadata.csv.gz",
        tsnepacsv_path="data/nlmcxr_tsne_pa.csv.gz",
        transform=transforms, data_aug=data_aug)

RSNA_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
    imgpath="data/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
    csvpath="data/kaggle-pneumonia-jpg/stage_2_train_labels.csv",
    dicomcsvpath="data/kaggle_stage_2_train_images_dicom_headers.csv.gz",
    transform=transforms, data_aug=data_aug, unique_patients=False)


################################### Dataset selection for Train, Validation and Inference  ##################################
############### Split 0 ##################### 
if cfg.split == 0:
    train_datas = [CHEX_dataset, MIMIC_CH_dataset]
    if "nih" in cfg.valid_data:
        valid_data = NIH_dataset
    if "pc" in cfg.valid_data:
        valid_data = PC_dataset
    
    test_data = GOOGLE_dataset
    test_data.data_aug = None
    
    xrv.datasets.default_pathologies = ['Pneumothorax', 'Fracture']
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, PC_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, NIH_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, CHEX_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, GOOGLE_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, MIMIC_CH_dataset)
    print(f"\n Common pathologies among all train and validation datasets: {xrv.datasets.default_pathologies}")
    
############### Split 1 #####################         
if cfg.split == 1:
    train_datas = [CHEX_dataset, MIMIC_CH_dataset]
    if "nih" in cfg.valid_data:
        valid_data = NIH_dataset
    if "pc" in cfg.valid_data:
        valid_data = PC_dataset
        
    test_data = OPENI_dataset
    test_data.data_aug = None
    
    xrv.datasets.default_pathologies = ['Effusion', 'Cardiomegaly', 'Edema']
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, PC_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, NIH_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, CHEX_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, OPENI_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, MIMIC_CH_dataset)       
    print(f"\n Common pathologies among all train and validation datasets: {xrv.datasets.default_pathologies}")
    
############### Split 2 #####################     
if cfg.split == 2:
    if "mc" in cfg.valid_data:
        train_datas = [NIH_dataset, CHEX_dataset] 
        valid_data = MIMIC_CH_dataset
    if "cx" in cfg.valid_data:
        train_datas = [NIH_dataset, MIMIC_CH_dataset] 
        valid_data = CHEX_dataset
        
    test_data = RSNA_dataset
    test_data.data_aug = None
    
    xrv.datasets.default_pathologies = ['Lung Opacity', 'Pneumonia']
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, NIH_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, CHEX_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, RSNA_dataset)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, MIMIC_CH_dataset)        
    print(f"\n Common pathologies among all train and validation datasets: {xrv.datasets.default_pathologies}")
    
    
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

def worker_init_fn(worker_id):
    np.random.seed(cfg.seed + worker_id)
    

train_loaders = [[{} for i in range(cfg.num_batches)] for i in range(len(train_datas))]
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


############################### Model, Optimizer and Criterion ################                   
model = xrv.models.DenseNet(num_classes=len(xrv.datasets.default_pathologies), in_channels=1, drop_rate=0.1, 
                                **xrv.models.get_densenet_params(cfg.model_name))

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, amsgrad=True)
criterion = torch.nn.BCEWithLogitsLoss()
 

########################################################### Train Function ###################################################
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

        train_nll = torch.stack([train_loaders[0][step]['loss'], train_loaders[1][step]['loss']]).mean()

        if cfg.cuda:
            weight_norm = torch.as_tensor(0., device=device)
        else:
            weight_norm = torch.as_tensor(0.)

        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss1 = train_loaders[0][step]['loss']
        loss2 = train_loaders[1][step]['loss']

        ### early loss mean
        loss1 = loss1.mean()
        loss2 = loss2.mean()

        loss = 0.0
        loss += cfg.erm_amount * (loss1 + loss2)

        loss += cfg.l2_regularizer_weight * weight_norm
        
        if not cfg.baseline:
#             print("Applying Penalty Weight ")        # Test if REx is running
            
            penalty_weight = (cfg.penalty_weight if step >= penalty_anneal_iters else 1.0)
            
            rex_penalty = torch.std(torch.as_tensor([[loss1, loss2]]), unbiased=False)
            
            loss += penalty_weight * rex_penalty
              
            if penalty_weight > 1.0:
                loss /= penalty_weight
        
        loss.backward()

        avg_loss.append(train_nll.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')
        
        optimizer.step()
    
    return np.mean(avg_loss)


################################################### Valid and Test Functions ################################################
def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
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
            
            if epoch is not None:
                t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')              
            
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
    
    if epoch is not None:
        print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')
    else:
        print(f'{name} - Avg AUC = {auc:4.4f}')

    return auc, np.mean(avg_loss), task_aucs


######################################################## Train ########################################################
def train(model, num_epochs):
    dataset_name = cfg.model_name + "-" + "valid"

    print(f'Using device: {device}')
    print(f"Output directory: {output_dir}")

    if not exists(output_dir):
        os.makedirs(output_dir)
    
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
    weights_files = glob(join(output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
        
    for epoch in range(start_epoch, num_epochs):
        np.random.seed(cfg.seed + epoch)
        train_loss = train_epoch(num_batches=cfg.num_batches,
                                 model=model,
                                 train_loaders=train_loaders,
                                 epoch=epoch,
                                 device=device,
                                 optimizer=optimizer)
        
        valid_auc, valid_loss, _ = valid_test_epoch(name='Valid',
                                             epoch=epoch,
                                             model=model,
                                             device=device,
                                             data_loader=valid_loader,
                                             criterion=criterion,
                                             limit=cfg.num_batches)
        
        
        tv_filename = output_dir.strip("/") + "_trainVal_results.csv"

        if os.path.exists(tv_filename):
            with open(tv_filename, 'a+') as csvfile:
                field_names = ['Epoch', 'Train Loss', 'Valid Loss', 'Valid AUC']
                writer = csv.DictWriter(csvfile, fieldnames=field_names)

                writer.writerow({'Epoch': epoch +1,
                                'Train Loss': round(train_loss, 4), 
                                 'Valid Loss': round(valid_loss, 4), 
                                 'Valid AUC': round(valid_auc, 4),
                                 })
        else:
            with open(tv_filename, 'w') as csvfile:
                field_names = ['Epoch', 'Train Loss', 'Valid Loss', 'Valid AUC']
                writer = csv.DictWriter(csvfile, fieldnames=field_names)

                writer.writeheader()

                writer.writerow({'Epoch': epoch +1,
                                 'Train Loss': round(train_loss, 4), 
                                 'Valid Loss': round(valid_loss, 4), 
                                 'Valid AUC': round(valid_auc, 4),
                                 })
        
        if np.mean(valid_auc) > best_metric:
            best_metric = np.mean(valid_auc)
            weights_for_best_validauc = model.state_dict()
            optimizer_for_best_validauc = optimizer.state_dict()
            torch.save(model, join(output_dir, f'{dataset_name}-best.pt'))
            

        stat = {
            "epoch": epoch + 1,
            "trainloss": train_loss,
            "validauc": valid_auc,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        torch.save(model, join(output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

    return metrics, best_metric,

metrics, best_metric, = train(model, num_epochs=cfg.num_epochs)
print(f"Best validation AUC: {best_metric:4.4f}")


###################################### Test ######################################
best_valid_state = output_dir + 'densenet121-valid-best.pt' 
model.load_state_dict(torch.load(best_valid_state).state_dict())
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()

test_auc, test_loss, task_aucs = valid_test_epoch(name='Test', 
                                                  epoch=None, 
                                                  model=model, 
                                                  device=device, 
                                                  data_loader=test_loader, 
                                                  criterion=criterion,
                                                  limit=cfg.num_batches)

print(f"Average AUC for all pathologies {test_auc:4.4f}")
print(f"Test loss: {test_loss:4.4f}")                                 
print(f"AUC for each task {[round(x, 4) for x in task_aucs]}")

test_filename = output_dir.strip("/") + "_test_results.csv"
            
with open(test_filename, 'w') as csvfile:
    if cfg.split == 0:
        field_names = ['Test_loss', 'Test_AVG_AUC', 'Pneumothorax', 'Fracture']
        
    elif cfg.split == 1:
        field_names = ['Test_loss', 'Test_AVG_AUC', 'Effusion', 'Cardiomegaly', 'Edema']
    else:
        field_names = ['Test_loss', 'Test_AVG_AUC', 'Lung Opacity', 'Pneumonia']
    
              
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    
    writer.writeheader()
    writer.writerow({field_names[0]: round(test_loss, 4), 
                     field_names[1]: round(test_auc, 4), 
                     field_names[2]: round(task_aucs[0], 4),
                     field_names[3]: round(task_aucs[1], 4),
                     field_names[-1]: round(task_aucs[-1], 4) if cfg.split==1 else None
                    })
