# OoD_Gen-Chest_Xray
Out-of-Distribution Generalization of Chest X-ray Using Risk Extrapolation

## Requirements (Installations)
Install the following libraries/packages with `pip`
```
torch 
torchvision
torchxrayvsion
```
## Four (4) Pathologies, Four (4) Datasets, & 12-Fold Cross-Validation
There are 12 different training, validation and test settings generated by combining 4 different Chest X-ray datasets (NIH ChestX-ray8 dataset, PadChest dataset, CheXpert, and MIMIC-CXR). These 12 settings are broken down into 6 splits (ranging from 0 to 5) that can be called by passing the argument `--split=<split>`. For each split, you have the option to choose between 2 validation datasets by passing the argument `--valid_data=<name of valid dataset>`. 
The dataset names are condensed as short strings: `"nih"`= NIH ChestX-ray8 dataset, `"pc"` = PadChest dataset, `"cx"` = CheXpert, and `"mc"` = MIMIC-CXR. \
For each setting, we compute the ROC-AUC for the following chest x-ray pathologies (labels): Cardiomegaly, Pneumonia, Effusion, Edema, Atelectasis, Consolidation, and Pneumothorax.

For each split, you train on two (2) datasets, validate on one (1) and test on the remaining one (1). \
The [chest.py](https://github.com/etetteh/OoD_Gen-Chest_Xray/blob/main/chest.py) file contains code to run the models in this study.

To **finetune** or perform **feature extraction** with ImageNet weights pass the `--pretrained` and `--feat_extract` arguments **respectively**

### Train Using Baseline Model (Merged Datasets)
To train a DenseNet-121 **Baseline** model by fine-tuning on the first split, and validate on the MIMIC-CXR dataset, with seed=0 run the following code:
```
python chest.py --merge_train --arch densenet121 --pretrained --weight_decay=0.0 --split 0 --valid_data mc --seed 0
```
Note that for the first split, PadChest is automatically selected as the `test_data`, when you pass MIMIC-CXR as the validation data, and vice versa.

### Train Balanced Mini-Batch Sampling
To train a DenseNet-121 **Balanced Mini-Batch Sampling** model by fine-tuning on the first split, and validate on the MIMIC-CXR dataset, with seed=0 run the following code:
```
python chest.py --arch densenet121 --pretrained --weight_decay=0.0 --split 0 --valid_data mc --seed 0
```
and always pass `--weight_decay=0.0` 

If no model architecture is specified, the code trains all the following architectures: `resnet50`, and `densenet121`.

### Inference using the XRV model
To perform inference using the DenseNet model with pretrained weights from [torchxrayvision](https://github.com/mlmed/torchxrayvision), run the following line of code:
```
python xrv_test.py --dataset_name pc --seed 0
```
Note that you can pass any of the arguments `pc`, `mc`, `cx` or `nih` to `--dataset_name` to run inference on PadChest, MIMIC-CXR, CheXpert and ChestX-Ray8 respectively. 
