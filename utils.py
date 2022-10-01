import os
import csv
import torchxrayvision as xrv
import torch, torchvision, torchvision.transforms


def write_results(filename: str, field_names: list, results: dict):
    read_file = open(filename, "r")
    results_file = csv.DictReader(read_file)
    update = []
    new = []
    row = {}
    for r in results_file:
        if r["Model"] == results["Model"]:
            for key, value in results.items():
                row[key] = value
            update = row
        else:
            for key, value in results.items():
                row[key] = value
            new = row
    
    read_file.close()

    if update:
        print("Results exists. Updating results in file...")
        print(update)
        update_file = open(filename, "w", newline='')
        data = csv.DictWriter(update_file, delimiter=',', fieldnames=field_names)
        data.writeheader()
        data.writerows([update])
    else:
        print("Results does not exist. Writing results to file...")
        print(new)
        update_file = open(filename, "a+", newline='')
        data = csv.DictWriter(update_file, delimiter=',', fieldnames=field_names)
        data.writerows([new])
    
    update_file.close()


def load_data(cfg):
    data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(
                cfg.data_aug_rot, 
                translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)
            ),                                    
            torchvision.transforms.ToTensor()
            ])

    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(cfg.data_resize)])

    ### Load NIH Dataset ### 
    NIH_dataset = None
    if "nih" == cfg.val_data or "nih" in cfg.train_datas:
        NIH_dataset = xrv.datasets.NIH_Dataset(
                imgpath=cfg.dataset_dir + "/images-224-NIH", 
                csvpath=cfg.dataset_dir + "/Data_Entry_2017_v2020.csv.gz",
                bbox_list_path=cfg.dataset_dir + "/BBox_List_2017.csv.gz",
                transform=transforms, data_aug=data_aug, unique_patients=False)
        xrv.datasets.relabel_dataset(cfg.pathologies, NIH_dataset)

    ## Load CHEXPERT Dataset ###
    CHEX_dataset = None
    if "cx" == cfg.val_data or "cx" in cfg.train_datas:
        CHEX_dataset = xrv.datasets.CheX_Dataset(
                imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
                csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
                transform=transforms, data_aug=data_aug, unique_patients=False)
        xrv.datasets.relabel_dataset(cfg.pathologies, CHEX_dataset)

    # ### Load MIMIC_CH Dataset ###
    MIMIC_CH_dataset = None
    if "mc" == cfg.val_data or "mc" in cfg.train_datas:
        MIMIC_CH_dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-MIMIC/files",
            csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
            metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
            transform=transforms, data_aug=data_aug, unique_patients=False)
        xrv.datasets.relabel_dataset(cfg.pathologies, MIMIC_CH_dataset)

    ### Load PADCHEST Dataset ###
    PC_dataset = None
    if "pc" == cfg.val_data or "pc" in cfg.train_datas:
        PC_dataset = xrv.datasets.PC_Dataset(
                imgpath=cfg.dataset_dir + "/PC/images-224",
                csvpath=cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                transform=transforms, data_aug=data_aug, unique_patients=False)
        xrv.datasets.relabel_dataset(cfg.pathologies, PC_dataset)


    print(f"\nCommon pathologies among all train and validation datasets: {cfg.pathologies}")


    datasets = {
        "nih": NIH_dataset,
        "cx": CHEX_dataset,
        "mc": MIMIC_CH_dataset,
        "pc": PC_dataset,
    }

    return datasets


def load_inference_data(cfg):
    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(cfg.data_resize)])
    
    if "nih" in cfg.test_data:
        ### Load NIH Dataset ### 
        NIH_dataset = xrv.datasets.NIH_Dataset(
                imgpath=cfg.dataset_dir + "/images-224-NIH", 
                csvpath=cfg.dataset_dir + "/Data_Entry_2017_v2020.csv.gz",
                bbox_list_path=cfg.dataset_dir + "/BBox_List_2017.csv.gz",
                transform=transforms, data_aug=None, unique_patients=False)
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, NIH_dataset)
        test_data = NIH_dataset

    if "mc" in cfg.test_data:
        # ### Load MIMIC_CH Dataset ###
        MIMIC_CH_dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=cfg.dataset_dir + "/images-224-MIMIC/files",
            csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
            metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
            transform=transforms, data_aug=None, unique_patients=False)
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, MIMIC_CH_dataset)
        test_data = MIMIC_CH_dataset 

    if "cx" in cfg.test_data:
        ## Load CHEXPERT Dataset ###
        CHEX_dataset = xrv.datasets.CheX_Dataset(
                imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
                csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
                transform=transforms, data_aug=None, unique_patients=False)
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, CHEX_dataset)
        test_data = CHEX_dataset

    if "pc" in cfg.test_data:
        ### Load PADCHEST Dataset ###
        PC_dataset = xrv.datasets.PC_Dataset(
                imgpath=cfg.dataset_dir + "/PC/images-224",
                csvpath=cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                transform=transforms, data_aug=None, unique_patients=False)
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, PC_dataset)
        test_data = PC_dataset

    if "gg" in cfg.test_data:
        ### Load GOOGLE Dataset ###
        GOOGLE_dataset = xrv.datasets.NIH_Google_Dataset(
                imgpath=cfg.dataset_dir + "/images-224-NIH",
                csvpath=cfg.dataset_dir + "/google2019_nih-chest-xray-labels.csv.gz",
                transform=transforms, data_aug=None
                )
        xrv.datasets.default_pathologies = ['Pneumothorax', 'Fracture']
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, GOOGLE_dataset)
        test_data = GOOGLE_dataset

    if "op" in cfg.test_data:
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

    if "rs" in cfg.test_data:    
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
    
    return test_data


def create_q_model(cfg, model):
    num_ftrs = model.fc.in_features
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
                    torch.nn.Linear(num_ftrs, cfg.num_labels)
               )

    new_model = torch.nn.Sequential(
                    model_features,
                    torch.nn.Flatten(1),
                    new_head,
                )
    return new_model

def create_model(cfg, model):
    if "resnet" in str(model.__class__):
        num_ftrs = model.fc.in_features
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Sequential(
                    torch.nn.Dropout(p=cfg.dropout),
                    torch.nn.Linear(num_ftrs, cfg.num_labels)
               )                       
        
    elif "densenet" in str(model.__class__):
        num_ftrs = model.classifier.in_features
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=cfg.dropout),
                    torch.nn.Linear(num_ftrs, cfg.num_labels)
               )

    return model

