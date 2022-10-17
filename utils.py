import os
import csv
import hashlib
import torchxrayvision as xrv
import torch
import torchvision
import torchvision.transforms


def get_cache_dir(data_dir, data_name):
    h = hashlib.sha1(data_dir.encode()).hexdigest()
    cache_dir = os.path.join("~", ".cache", "torchxrayvision", "datasets", f"{data_name}", h[:17] + ".pt")
    cache_dir = os.path.expanduser(cache_dir)

    return cache_dir


def load_data(cfg):
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(
            cfg.data_aug_rot,
            translate=(cfg.data_aug_trans, cfg.data_aug_trans),
            scale=(1.0 - cfg.data_aug_scale, 1.0 + cfg.data_aug_scale)
        ),
        torchvision.transforms.ToTensor()
    ])

    transforms = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(cfg.data_resize)
    ])

    # Load and cache NIH Dataset ###
    nih_dataset = None
    if "nih" in cfg.train_datas or "nih" in cfg.val_data or "nih" in cfg.test_data:
        imgdir = os.path.join(cfg.dataset_dir, "images-224-NIH")
        cache_dir = get_cache_dir(imgdir, "nih")
        if cfg.cache_dataset and os.path.exists(cache_dir):
            nih_dataset, _ = torch.load(cache_dir)
        else:
            nih_dataset = xrv.datasets.NIH_Dataset(
                imgpath=imgdir,
                csvpath=os.path.join(cfg.dataset_dir, "Data_Entry_2017_v2020.csv.gz"),
                bbox_list_path=os.path.join(cfg.dataset_dir, "BBox_List_2017.csv.gz"),
                transform=transforms,
                data_aug=None if cfg.test_data == "nih" else data_aug,
                unique_patients=False
            )
            xrv.datasets.relabel_dataset(cfg.pathologies, nih_dataset)
            if cfg.cache_dataset:
                os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
                torch.save((nih_dataset, imgdir), cache_dir)

    # Load and cache CHEXPERT Dataset ###
    cx_dataset = None
    if "cx" in cfg.train_datas or "cx" in cfg.val_data or "cx" in cfg.test_data:
        imgdir = os.path.join(cfg.dataset_dir, "CheXpert-v1.0-small")
        cache_dir = get_cache_dir(imgdir, "cx")
        if cfg.cache_dataset and os.path.exists(cache_dir):
            cx_dataset, _ = torch.load(cache_dir)
        else:
            cx_dataset = xrv.datasets.CheX_Dataset(
                imgpath=imgdir,
                csvpath=os.path.join(cfg.dataset_dir, "CheXpert-v1.0-small/train.csv"),
                transform=transforms,
                data_aug=None if cfg.test_data == "cx" else data_aug,
                unique_patients=False
            )
            xrv.datasets.relabel_dataset(cfg.pathologies, cx_dataset)
            if cfg.cache_dataset:
                os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
                torch.save((cx_dataset, imgdir), cache_dir)

    # Load and cache MIMIC_CH Dataset ###
    mc_dataset = None
    if "mc" in cfg.train_datas or "mc" in cfg.val_data or "mc" in cfg.test_data:
        imgdir = os.path.join(cfg.dataset_dir, "images-224-MIMIC/files")
        cache_dir = get_cache_dir(imgdir, "mc")
        if cfg.cache_dataset and os.path.exists(cache_dir):
            mc_dataset, _ = torch.load(cache_dir)
        else:
            mc_dataset = xrv.datasets.MIMIC_Dataset(
                imgpath=imgdir,
                csvpath=os.path.join(cfg.dataset_dir, "MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz"),
                metacsvpath=os.path.join(cfg.dataset_dir, "MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz"),
                transform=transforms,
                data_aug=None if cfg.test_data == "mc" else data_aug,
                unique_patients=False
            )
            xrv.datasets.relabel_dataset(cfg.pathologies, mc_dataset)
            if cfg.cache_dataset:
                os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
                torch.save((mc_dataset, imgdir), cache_dir)

    # Load and cache PADCHEST Dataset ###
    pc_dataset = None
    if "pc" in cfg.train_datas or "pc" in cfg.val_data or "pc" in cfg.test_data:
        imgdir = os.path.join(cfg.dataset_dir, "PC/images-224")
        cache_dir = get_cache_dir(imgdir, "pc")
        if cfg.cache_dataset and os.path.exists(cache_dir):
            pc_dataset, _ = torch.load(cache_dir)
        else:
            pc_dataset = xrv.datasets.PC_Dataset(
                imgpath=imgdir,
                csvpath=os.path.join(cfg.dataset_dir, "PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"),
                transform=transforms,
                data_aug=None if cfg.test_data == "pc" else data_aug,
                unique_patients=False
            )
            xrv.datasets.relabel_dataset(cfg.pathologies, pc_dataset)
            if cfg.cache_dataset:
                os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
                torch.save((pc_dataset, imgdir), cache_dir)

    print(f"\nCommon pathologies among all train and validation datasets: {cfg.pathologies}")

    datasets = {
        "nih": nih_dataset,
        "cx": cx_dataset,
        "mc": mc_dataset,
        "pc": pc_dataset,
    }

    return datasets


def create_q_model(cfg, model):
    num_features = model.fc.in_features
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_features = torch.nn.Sequential(
        model.quant,
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        model.dequant,
    )

    new_head = torch.nn.Sequential(
        torch.nn.Dropout(p=cfg.dropout),
        torch.nn.Linear(num_features, cfg.num_labels)
    )

    new_model = torch.nn.Sequential(
        model_features,
        torch.nn.Flatten(1),
        new_head,
    )

    return new_model


def create_model(cfg, model):
    if "resnet" in str(model.__class__):
        num_features = model.fc.in_features
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(num_features, cfg.num_labels)
        )

    elif "densenet" in str(model.__class__):
        num_features = model.classifier.in_features
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(num_features, cfg.num_labels)
        )

    return model
