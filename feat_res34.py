import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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

import torchvision, torchvision.transforms
import skimage.transform
import sklearn.metrics
import sklearn, sklearn.model_selection
from sklearn.metrics import roc_auc_score, accuracy_score

import torchxrayvision.models as models
import torchxrayvision.datasets as datasets
from agc import adaptive_clip_grad      #https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
from pytorchtools import EarlyStopping  #https://github.com/Bjarten/early-stopping-pytorch
from madgrad import MADGRAD             #https://github.com/facebookresearch/madgrad

parser = argparse.ArgumentParser(description='X-RAY Pathology Detection - Using Risk Extrapolation')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--name', type=str, default="valid")
parser.add_argument('--output_dir', type=str, default="rex_feat/")
parser.add_argument('--model_name', type=str, default="resnet18")
parser.add_argument('--dataset_name', type=str, default="nih_mc_cx")
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--num_epochs', type=int, default=250, help='')  
parser.add_argument('--patience', type=int, default=10, help='')

### Data loader
parser.add_argument('--cuda', type=bool, default=False, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--shuffle', type=bool, default=False, help='')
parser.add_argument('--threads', type=int, default=16, help='')
parser.add_argument('--batches', type=int, default=1000, help='')

### Data Augmentation                  
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')

### REx hyperparameters
parser.add_argument('--erm_amount', type=float, default=1.0)
parser.add_argument('--l2_regularizer_weight', type=float, default=1e-5) 
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)

cfg = parser.parse_args()
print(cfg)

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

cuda = False
device = 'cuda' if cfg.cuda else 'cpu'
if not torch.cuda.is_available() and cfg.cuda:
    device = 'cpu'
    print("WARNING: cuda was requested but is not available, using cpu instead.")
print(f'Using device: {device}')

###################################################################################
data_aug = torchvision.transforms.Compose([
        datasets.ToPILImage(),
#         torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
                                            #resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor()
    ])
print(data_aug)
transforms = torchvision.transforms.Compose([datasets.XRayCenterCrop(), datasets.XRayResizer(224)])

datas = []
datas_names = []

### Load NIH Dataset ### 
NIH_dataset = datasets.NIH_Dataset(
        imgpath="images-224-NIH", 
        csvpath="Data_Entry_2017_v2020.csv.gz",
        bbox_list_path="BBox_List_2017.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False)
datas.append(NIH_dataset)
datas_names.append("nih")

# ### Load MIMIC_CH Dataset ###
MIMIC_CH_dataset = datasets.MIMIC_Dataset(
    imgpath="images-224-MIMIC/files",
    csvpath="MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
    metacsvpath="MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
    transform=transforms, data_aug=data_aug, unique_patients=False)
datas.append(MIMIC_CH_dataset)
datas_names.append("mimic_ch")

### Load CHEXPERT Dataset ###
CHEX_dataset = datasets.CheX_Dataset(
         imgpath="CheXpert-v1.0-small",
         csvpath="CheXpert-v1.0-small/train.csv",
         transform=transforms, data_aug=data_aug, unique_patients=False)

### Load PADCHEST Dataset ###
PC_dataset = datasets.PC_Dataset(
        imgpath="PC/images-224",
        csvpath="PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)
datas.append(PC_dataset)
datas_names.append("pc")

# nh = NIH_dataset.pathologies
# mc = MIMIC_CH_dataset.pathologies
# cx = CHEX_dataset.pathologies
# pc = PC_dataset.pathologies

### create list of all datasets
# pat = [nh, mc, cx, pc]

## Find pathologies comman to all datasets
# intersection = list(set.intersection(*map(set, pat)))
# datasets.default_pathologies = intersection
# datasets.default_pathologies

datasets.default_pathologies = ['Effusion', 'Consolidation', 'Cardiomegaly', 'Pneumonia']
datasets.relabel_dataset(datasets.default_pathologies, NIH_dataset)
datasets.relabel_dataset(datasets.default_pathologies, CHEX_dataset)
datasets.relabel_dataset(datasets.default_pathologies, MIMIC_CH_dataset)
datasets.relabel_dataset(datasets.default_pathologies, PC_dataset)

train_loaders = []
for edx, data in enumerate(datas):
    train_loader = torch.utils.data.DataLoader(data,
                                       batch_size=cfg.batch_size,
                                       shuffle=cfg.shuffle,
                                       num_workers=cfg.threads, 
                                       pin_memory=True,
                                       drop_last=True)
    train_loaders.append(train_loader)

valid_loader = torch.utils.data.DataLoader(CHEX_dataset,
                               batch_size=cfg.batch_size,
                               shuffle=cfg.shuffle,
                               num_workers=cfg.threads, 
                               pin_memory=True,
                               drop_last=True)

nih = []
for batch_idx, samples in enumerate(tqdm(train_loaders[0])):
    if batch_idx >= cfg.batches:
        print("breaking out")
        break
    nih.append({"images": samples["img"].float(), "targets": samples["lab"]})
nih.append({"task_weights": train_loaders[0].dataset.labels})

mimic_ch = []
for batch_idx, samples in enumerate(tqdm(train_loaders[1])):   
    if batch_idx >= cfg.batches:
        print("breaking out")
        break    
    mimic_ch.append({"images": samples["img"].float(), "targets": samples["lab"]})
mimic_ch.append({"task_weights": train_loaders[1].dataset.labels})

pc = []
for batch_idx, samples in enumerate(tqdm(train_loaders[2])):   
    if batch_idx >= cfg.batches:
        print("breaking out")
        break    
    pc.append({"images": samples["img"].float(), "targets": samples["lab"]})
pc.append({"task_weights": train_loaders[2].dataset.labels})

# pc = []
# for batch_idx, samples in enumerate(tqdm(train_loaders[2])):
#     if batch_idx >= cfg.batches:
#         print("breaking out")
#         break
#     pc.append({"images": samples["img"].float(), "targets": samples["lab"]})
# pc.append({"task_weights": train_loaders[2].dataset.labels})

envs = [nih, mimic_ch, pc]

###################################################################################
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(datasets.default_pathologies))
model = model.to(device)

print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

optimizer = MADGRAD(params_to_update, lr=cfg.lr)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, epochs=cfg.num_epochs, steps_per_epoch=cfg.batches, three_phase=True)

#model = models.DenseNet(num_classes=len(intersection), in_channels=1, 
#                                **models.get_densenet_params(cfg.model_name)

###################################################################################
def train(outputs, targets, tw, device):   
    weights = np.nansum(tw, axis=0)
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
    
    loss = loss.sum()                
    
    return loss

def train_epoch(cfg, model, envs, epoch, device, optimizer):
    #scaler = torch.cuda.amp.GradScaler()
    model.train()
    avg_loss = []
    t = tqdm(range(cfg.batches))
    for i in t:
        for edx, env in enumerate(envs):
            #with torch.cuda.amp.autocast():
            outputs = model(env[i]["images"])
            env[i]['loss'] = train(outputs=outputs, targets=env[i]["targets"], tw=env[-1]["task_weights"], device=device)

        train_nll = torch.stack([envs[0][i]['loss'], envs[1][i]['loss'], envs[2][i]['loss']]).mean()

        if cfg.cuda:
            weight_norm = torch.tensor(0., device=torch.device('cuda:0'))
        else:
            weight_norm = torch.tensor(0.)
            
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss1 = envs[0][i]['loss']
        loss2 = envs[1][i]['loss']
        loss3 = envs[2][i]['loss']
        
        ### early loss mean
        loss1 = loss1.mean()
        loss2 = loss2.mean()
        loss3 = loss3.mean()
        
        loss = 0.0
        loss += cfg.erm_amount * (loss1 + loss2 + loss3)

        loss += cfg.l2_regularizer_weight * weight_norm

        penalty_weight = (cfg.penalty_weight 
          if epoch >= cfg.penalty_anneal_iters else 1.0)
        
        rex_penalty = torch.std(torch.Tensor([[loss1, loss2, loss3]]), unbiased=False)

        #rex
        loss += penalty_weight * rex_penalty

        if penalty_weight > 1.0:
          # Rescale the entire loss to keep gradients in a reasonable range
          loss /= penalty_weight
        
        #scaler.scale(loss).backward()
        optimizer.zero_grad()
        loss.backward()
        adaptive_clip_grad(model.parameters(), clip_factor=0.01, eps=1e-3, norm_type=2.0)
        
        avg_loss.append(train_nll.detach())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {torch.Tensor(avg_loss).mean():4.4f}')

        #scaler.step(optimizer)
        #scaler.update()
        optimizer.step()
        scheduler.step()

    return torch.Tensor(avg_loss).mean()

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=cfg.batches//2):
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
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {torch.Tensor(avg_loss).mean():4.4f}')
            
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

    return auc, torch.Tensor(avg_loss).mean()

###################################################################################

def rex_train(model, num_epochs, patience):        
    dataset_name = cfg.dataset_name + "-" + cfg.model_name + "-" + cfg.name

    print(f'Using device: {device}')

    print(cfg.output_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


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
     
    for epoch in range(start_epoch, num_epochs):

        avg_loss = train_epoch(cfg=cfg,
                               model=model,
                               envs=envs,
                               epoch=epoch,
                               device=device,
                               optimizer=optimizer)
        
        auc_valid, loss_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion)


        train_losses.append(avg_loss) 
        valid_losses.append(loss_valid)

        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            optimizer_for_best_validauc = optimizer.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        torch.save(model, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))
        
        early_stopping(loss_valid, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return metrics, best_metric, train_losses, valid_losses    

metrics, best_metric, train_losses, valid_losses = rex_train(model, num_epochs=cfg.num_epochs, patience=cfg.patience)

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(valid_losses)+1),valid_losses,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_losses.index(min(valid_losses))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.title("Risk Extrapolation: Effusion and Consolidation (ResNet18)")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0.5, 5.0) # consistent scale
plt.xlim(0, len(train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('REx_ResNet18_loss_plot.png', bbox_inches='tight')
