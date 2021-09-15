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
from merger import Merge_Dataset


parser = argparse.ArgumentParser(description='X-RAY Pathology Detection')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--dataset_dir', type=str, default="./data/")
parser.add_argument('--ssl_ckpt_dir', type=str, default="")
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='')

parser.add_argument('--split', type=int, default=0, help='')
parser.add_argument('--valid_data', type=str, default="mc", help='')

parser.add_argument('--baseline', action='store_true', default=False)
parser.add_argument('--pretrained', action='store_true', default=False, help='')
parser.add_argument('--feat_extract', action='store_true', default=False, help='')
parser.add_argument('--merge_train', action='store_true', default=False, help='')
parser.add_argument('--arch', nargs="+", default=None, help='Architecture of model')

### Data loader
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--num_batches', type=int, default=430, help='')
parser.add_argument('--num_epochs', type=int, default=200, help='')

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
        imgpath=cfg.dataset_dir + "/images-224-NIH", 
        csvpath=cfg.dataset_dir + "/Data_Entry_2017_v2020.csv.gz",
        bbox_list_path=cfg.dataset_dir + "/BBox_List_2017.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False)

## Load CHEXPERT Dataset ###
CHEX_dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)

# ### Load MIMIC_CH Dataset ###
MIMIC_CH_dataset = xrv.datasets.MIMIC_Dataset(
    imgpath=cfg.dataset_dir + "/images-224-MIMIC/files",
    csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
    metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
    transform=transforms, data_aug=data_aug, unique_patients=False)

### Load PADCHEST Dataset ###
PC_dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/PC/images-224",
        csvpath=cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)

xrv.datasets.default_pathologies = ['Cardiomegaly',
                                     'Effusion',
                                     'Edema',
                                     'Consolidation',
                                     ]

print(f"Common pathologies among all train and validation datasets: {xrv.datasets.default_pathologies}")

xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, NIH_dataset)
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, MIMIC_CH_dataset)
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, CHEX_dataset)
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, PC_dataset)

################################### Dataset selection for Train, Validation and Inference  ##################################
############### Split 0 ##################### 
if cfg.split == 0:
    train_datas = [NIH_dataset, CHEX_dataset]
    if "mc" in cfg.valid_data:
        valid_data = MIMIC_CH_dataset
        test_data = PC_dataset
        test_data.data_aug = None
    if "pc" in cfg.valid_data:
        valid_data = PC_dataset
        test_data = MIMIC_CH_dataset
        test_data.data_aug = None

############### Split 1 #####################         
if cfg.split == 1:
    train_datas = [NIH_dataset, PC_dataset]
    if "cx" in cfg.valid_data:
        valid_data = CHEX_dataset
        test_data = MIMIC_CH_dataset
        test_data.data_aug = None
    if "mc" in cfg.valid_data:
        valid_data = MIMIC_CH_dataset
        test_data = CHEX_dataset
        test_data.data_aug = None    

############### Split 2 #####################         
if cfg.split == 2:
    train_datas = [NIH_dataset, MIMIC_CH_dataset]
    if "pc" in cfg.valid_data:
        valid_data = PC_dataset
        test_data = CHEX_dataset
        test_data.data_aug = None
    if "cx" in cfg.valid_data:
        valid_data = CHEX_dataset
        test_data = PC_dataset
        test_data.data_aug = None    

################### Split 3 ##################### 
if cfg.split == 3:
    train_datas = [CHEX_dataset, MIMIC_CH_dataset]
    if "pc" in cfg.valid_data:
        valid_data = PC_dataset
        test_data = NIH_dataset 
        test_data.data_aug = None
    if "nih" in cfg.valid_data:
        valid_data = NIH_dataset
        test_data = PC_dataset
        test_data.data_aug = None   
    
################ Split 4 ###################     
if cfg.split == 4:
    train_datas = [CHEX_dataset, PC_dataset]
    if "mc" in cfg.valid_data:
        valid_data = MIMIC_CH_dataset
        test_data = NIH_dataset
        test_data.data_aug = None
    if "nih" in cfg.valid_data:
        valid_data = NIH_dataset
        test_data = MIMIC_CH_dataset
        test_data.data_aug = None  

################## Split 5 #####################         
if cfg.split == 5:
    train_datas = [MIMIC_CH_dataset, PC_dataset]
    if "cx" in cfg.valid_data:
        valid_data = CHEX_dataset
        test_data = NIH_dataset
        test_data.data_aug = None
    if "nih" in cfg.valid_data:
        valid_data = NIH_dataset
        test_data = CHEX_dataset
        test_data.data_aug = None        

np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

if cfg.merge_train:
    cmerge = xrv.datasets.Merge_Dataset(train_datas)
    dmerge = xrv.datasets.Merge_Dataset(train_datas)
    train_datas = [cmerge, dmerge]
    
if cfg.baseline:
    train_loader = [[{} for i in range(2*cfg.num_batches)]]
    train_dataset = Merge_Dataset(datasets=train_datas, seed=cfg.seed, num_samples=cfg.batch_size*cfg.num_batches)
    dataloader = DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       shuffle=True,
                       num_workers=cfg.num_workers, 
                       pin_memory=True,
                       drop_last=True)
    train_loader[0].insert(0, dataloader)
else:   
    train_loader = [[{} for i in range(cfg.num_batches)] for i in range(len(train_datas))]
    for dataloader in train_loader:
        for data in train_datas:
            if train_loader.index(dataloader) == train_datas.index(data):
                tr_l = DataLoader(xrv.datasets.SubsetDataset(dataset=data, idxs=range(cfg.batch_size*cfg.num_batches)),
                               batch_size=cfg.batch_size,
                               shuffle=True,
                               num_workers=cfg.num_workers,
                               pin_memory=True,
                               drop_last=True)
                dataloader.insert(0, tr_l)

valid_loader = DataLoader(valid_data,
                       batch_size=cfg.batch_size,
                       shuffle=True,
                       num_workers=cfg.num_workers,
                       pin_memory=True,
                       drop_last=True)

test_loader = DataLoader(test_data,
                       batch_size=cfg.batch_size,
                       shuffle=False,
                       num_workers=cfg.num_workers,
                       pin_memory=True,
                       drop_last=False)


############################### Model, Optimizer and Criterion ################                   
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    
######################################################### Train Function ############################################
def compute_loss(outputs, targets, train_loader, criterion, device):    
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

def train_epoch(num_batches, epoch, model, device, train_loader, criterion, optimizer):
    model.train()
    avg_loss = []
    
    if cfg.baseline:
        t = tqdm(range(1, 2*num_batches+1))
    else:
        t = tqdm(range(1, num_batches+1))
        
    for step in t:
        for idx, dataloader in enumerate(train_loader):
            optimizer.zero_grad()
            
            dataloader_iterator = iter(dataloader[0])
            sample = next(dataloader_iterator)
            image, target = sample["img"].float().to(device), sample["lab"].to(device)
            
            outputs = model(image)

            dataloader[step]["loss"] = compute_loss(outputs, target, dataloader[0], criterion, device)

        if cfg.baseline:
            train_nll = torch.tensor(train_loader[0][step]['loss'])
            loss = 0.0
            loss1 = train_loader[0][step]['loss']
            loss += loss1
        else:
            train_nll = torch.stack([train_loader[0][step]['loss'], train_loader[1][step]['loss']]).mean()

            weight_norm = torch.tensor(0.).to(device)
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            loss1 = train_loader[0][step]['loss']
            loss2 = train_loader[1][step]['loss']

            loss = 0.0
            loss += (loss1 + loss2)
            loss += 1e-5 * weight_norm
        
        loss.backward()
        optimizer.step()
        
        avg_loss.append(train_nll.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')
        
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
            
            images = samples["img"].to(device)#.cuda()
            targets = samples["lab"].to(device)#.cuda()

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
def main(model, model_name, output_dir, num_epochs):
    dataset_name = model_name + "-" + "valid"

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


    params_to_update = model.parameters()
    if cfg.feat_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        print(f"Updating all parameters")
        

    # Optimizer
    optimizer = torch.optim.Adam(params_to_update, lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(output_dir, f'{dataset_name}-e*.pt'))
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
        train_loss = train_epoch(num_batches=cfg.num_batches,
                                        epoch=epoch, 
                                        model=model, 
                                        device=device, 
                                        train_loader=train_loader, 
                                        criterion=criterion, 
                                        optimizer=optimizer, 
                                        )
        
        valid_auc, valid_loss, _ = valid_test_epoch(
                                         name='Valid',
                                         epoch=epoch,
                                         model=model,
                                         device=device,
                                         data_loader=valid_loader,
                                         criterion=criterion,
                                         limit=cfg.num_batches//2.5)
        
        
        tv_filename = output_dir + output_dir.strip("/") + "_trainVal_results.csv"

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

def get_model_inputimg(model_name, num_classes):
    model = None
    input_size = 0
    
    if cfg.feat_extract and cfg.pretrained:
        print(f"\n\n=> Using pre-trained model {model_name} as feature extractor")
    elif cfg.pretrained and not cfg.feat_extract:
        print(f"\n\n=> Using pre-trained model {model_name} for finetuning")
    else:
        print(f"\n\n=> Creating model {model_name}")
    
    if "resnet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=cfg.pretrained)
        set_parameter_requires_grad(model, cfg.feat_extract)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        
    elif "densenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=cfg.pretrained)
        set_parameter_requires_grad(model, cfg.feat_extract)
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
    else:
        print(f"Invalid model name. Choose one of the models defined for this work")
        exit()
        
    return model


if cfg.arch is not None:
    model_zoo = cfg.arch
else:
    model_zoo = sorted(['resnet50', 'densenet121',])

for model_name in model_zoo:
    model = get_model_inputimg(model_name, num_classes=len(xrv.datasets.default_pathologies)) 
    model = model.to(device)

    if cfg.baseline:
        print("\n Training Baseline Model \n")
        output_dir = "baseline_split-" + str(cfg.split) + "_" + model_name + "_valid-" + cfg.valid_data + "/"
    else:
        print("\n Training REx Baseline Model \n")
        output_dir = "bmbs_split-" +  str(cfg.split) + "_" + model_name + "_valid-" + cfg.valid_data + "/"
        
    metrics, best_metric, = main(model, model_name, output_dir, num_epochs=cfg.num_epochs)
    print(f"Best validation AUC: {best_metric:4.4f}")


    ###################################### Test ######################################
    best_valid_state = output_dir + model_name + '-valid-best.pt'

    model.load_state_dict(torch.load(best_valid_state).state_dict())
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    test_auc, test_loss, task_aucs = valid_test_epoch(name='Test', 
                                                      epoch=None, 
                                                      model=model, 
                                                      device=device, 
                                                      data_loader=test_loader, 
                                                      criterion=criterion,
                                                      limit=cfg.num_batches//2)

    print(f"Average AUC for all pathologies {test_auc:4.4f}")
    print(f"Test loss: {test_loss:4.4f}")                                 
    print(f"AUC for each task {[round(x, 4) for x in task_aucs]}")

    test_filename = output_dir + output_dir.strip("/") + "_test_results.csv"

    with open(test_filename, 'w') as csvfile:
        field_names = ['Test_loss', 'Test_AVG_AUC',
                       'Cardiomegaly',
                       'Effusion',
                       'Edema',
                       'Consolidation',
                       ]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        writer.writeheader()
        writer.writerow({
                         'Test_loss': round(test_loss, 4), 
                         'Test_AVG_AUC': round(test_auc, 4), 
                         'Cardiomegaly': round(task_aucs[0], 4),
                         'Effusion': round(task_aucs[1], 4),
                         'Edema': round(task_aucs[2], 4),
                         'Consolidation': round(task_aucs[3], 4),
                        })
