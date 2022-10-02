import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import os, copy, time, datetime
import random
import argparse
import json
import sklearn

from tqdm import tqdm as tqdm_base
from sklearn.metrics import roc_auc_score

import wandb
import torch
import torchvision, torchvision.transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import utils
import numpy as np
import torchxrayvision as xrv


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def compute_loss(outputs, targets, train_loader, criterion, device):
    weights = np.nansum(train_loader.dataset.labels, axis=0)
    weights = weights.max() - weights + weights.mean()
    weights = weights / weights.max()
    weights = torch.from_numpy(weights).to(device).float()

    loss = torch.zeros(1).to(device).float()
    for task in range(targets.shape[1]):
        task_output = outputs[:, task]
        task_target = targets[:, task]
        mask = ~torch.isnan(task_target)
        task_output = task_output[mask]
        task_target = task_target[mask]
        if len(task_target) > 0:
            task_loss = criterion(task_output.float(), task_target.float())
            loss += weights[task] * task_loss
    return loss.sum()


def train_one_epoch(num_batches, epoch, model, device, train_loader, criterion, optimizer):
    model.train()
    avg_loss = []
    t = tqdm(range(1, num_batches + 1))

    for step in t:
        for idx, dataloader in enumerate(train_loader):
            optimizer.zero_grad()

            dataloader_iterator = iter(dataloader[0])
            sample = next(dataloader_iterator)
            image, target = sample["img"].float().to(device), sample["lab"].to(device)

            outputs = model(image)

            dataloader[step]["loss"] = compute_loss(outputs, target, dataloader[0], criterion, device)
        train_nll = torch.stack([train_loader[0][step]["loss"], train_loader[1][step]["loss"]]).mean()

        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss1 = train_loader[0][step]["loss"]
        loss2 = train_loader[1][step]["loss"]
        loss = 0.0
        loss += (loss1 + loss2)
        loss += 1e-5 * weight_norm

        loss.backward()
        optimizer.step()

        avg_loss.append(train_nll.detach().cpu().numpy())
        t.set_description(f"Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}")
    return np.mean(avg_loss)


def evaluate(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs = {}
    task_targets = {}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []

    with torch.inference_mode():
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
                task_output = outputs[:, task]
                task_target = targets[:, task]
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
                t.set_description(f"Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}")

        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])

        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task])) > 1:
                task_auc = roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)
    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])

    if epoch is not None:
        print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, np.mean(avg_loss), task_aucs


def main(cfg):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.device)
    output_dir = f"{cfg.arch}_mergetrain-{cfg.merge_train}_traindata-{'_'.join(cfg.train_datas)}_valdata-{cfg.val_data}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg.pathologies = ["Cardiomegaly", "Effusion", "Edema", "Consolidation"]
    wandb.log({"Pathologies": cfg.pathologies})

    cfg.num_labels = len(cfg.pathologies)

    model = torchvision.models.__dict__[cfg.arch](weights=cfg.weights)
    if cfg.feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    model = utils.create_model(cfg, model)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = torch.optim.Adam(params_to_update, lr=cfg.lr, weight_decay=0.0, amsgrad=True)

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.lr_warmup_epochs, eta_min=cfg.lr_min
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=cfg.lr_warmup_decay, total_iters=cfg.lr_warmup_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[cfg.lr_warmup_epochs]
    )

    best_metric = 0.0
    if os.path.isfile(os.path.join(output_dir, "checkpoint.pth")):
        checkpoint = torch.load(os.path.join(output_dir, "checkpoint.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        cfg.start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_auc"]
        results = checkpoint["best_task_aucs"]

    if cfg.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        test_data = utils.load_inference_data(cfg)
        test_loader = DataLoader(test_data,
                                 batch_size=cfg.batch_size,
                                 sampler=SequentialSampler(test_data),
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 )

        if os.path.isfile(os.path.join(output_dir, "best_model.pth")):
            state = torch.load(os.path.join(output_dir, "best_model.pth"),
                               map_location="cpu")
            model.load_state_dict(state)
        test_auc, test_loss, task_aucs = evaluate(
            name="Inference",
            model=model,
            epoch=None,
            device=device,
            data_loader=test_loader,
            criterion=criterion,
            limit=cfg.num_batches // 2
        )

        results = {"Test Avg AUC": round(test_auc, 4),
                   "Test Task AUCs": {"Cardiomegaly": round(task_aucs[0], 4),
                                      "Effusion": round(task_aucs[1], 4),
                                      "Edema": round(task_aucs[2], 4),
                                      "Consolidation": round(task_aucs[3], 4)
                                      }
                   }
        wandb.log({"Test results": results})
        print(json.dumps(results))
        return

    if not cfg.test_only:
        datasets = utils.load_data(cfg)
        train_datas = [datasets[data] for data in cfg.train_datas]
        valid_data = datasets[cfg.val_data]

        if cfg.merge_train:
            cmerge = xrv.datasets.Merge_Dataset(train_datas)
            dmerge = xrv.datasets.Merge_Dataset(train_datas)
            train_datas = [cmerge, dmerge]

        train_loader = [[{} for i in range(cfg.num_batches)] for i in range(len(train_datas))]
        for dataloader in train_loader:
            for data in train_datas:
                if train_loader.index(dataloader) == train_datas.index(data):
                    train_data = xrv.datasets.SubsetDataset(dataset=data, idxs=range(cfg.batch_size * cfg.num_batches))
                    tr_l = DataLoader(train_data,
                                      batch_size=cfg.batch_size,
                                      sampler=RandomSampler(train_data),
                                      num_workers=cfg.num_workers,
                                      pin_memory=True,
                                      )
                    dataloader.insert(0, tr_l)

        val_loader = DataLoader(valid_data,
                                batch_size=cfg.batch_size,
                                sampler=SequentialSampler(valid_data),
                                num_workers=cfg.num_workers,
                                pin_memory=True,
                                )

    print(f"\nOutput directory: {output_dir}")
    print(f"\nUsing device: {device}")

    wandb.watch(model)
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        train_loss = train_one_epoch(
            num_batches=cfg.num_batches,
            epoch=epoch,
            model=model,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        lr_scheduler.step()

        with torch.inference_mode():
            val_auc, val_loss, task_aucs = evaluate(
                name="Val",
                epoch=epoch,
                model=model,
                device=device,
                data_loader=val_loader,
                criterion=criterion,
                limit=cfg.num_batches // 2
            )

        if val_auc > best_metric:
            best_metric = val_auc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pth"))
            results = {"Best Val Task AUCs": {"Cardiomegaly": round(task_aucs[0], 4),
                                              "Effusion": round(task_aucs[1], 4),
                                              "Edema": round(task_aucs[2], 4),
                                              "Consolidation": round(task_aucs[3], 4)
                                              }
                       }
            wandb.log({"Val results": results})
            print(json.dumps(results))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_auc": best_metric,
            "best_task_aucs": results,
            "config": cfg,
        }
        torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pth"))
        torch.save(checkpoint, os.path.join(wandb.run.dir, "checkpoint.pth"))

        wandb.log({"Train Loss": train_loss})
        wandb.log({"Val Loss": val_loss})
        wandb.log({"Val AUC": val_auc})
        wandb.log({"Best Val AUC": best_metric})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print(f"Best validation AUC: {best_metric:4.4f}")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Chest X-RAY Pathology Classification", add_help=add_help)

    parser.add_argument("--arch", type=str, default="densenet121",
                        help="Model architecture name. One of [densenet121, resnet50]")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="The starting epoch. automatically assigned when resuming training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of passes through the whole dataset")
    parser.add_argument("--resume", type=str, help="A model checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=0, help="Seed for RNG")
    parser.add_argument("--merge_train", action="store_true",
                        help="Whether to merge train datasets (baseline) or not merge and sample mini batches from each set")

    parser.add_argument("--dataset_dir", type=str, default="./data/", help="Datasets directory")
    parser.add_argument("--train_datas", nargs="+",
                        help="List of training datasets. pass only two of ['cx', 'mc', 'nih', 'pc'] at a time")
    parser.add_argument("--val_data", type=str, default=" ",
                        help="validation dataset. Should be different from the train datas. One of ['cx', 'mc', 'nih', 'pc']")
    parser.add_argument("--test_data", type=str, default=" ", help="Test dataset. One of ['cx', 'mc', 'nih', 'pc']")
    parser.add_argument("--cache_dataset", action="store_true", help="Whether or not to cache the dataset")

    parser.add_argument("--weights", type=str, default="DEFAULT",
                        help="Pretrained weights toload PyTorch model. One of ['DEFAULT', 'None']")
    parser.add_argument("--feature_extract", action="store_true",
                        help="Whether to use the model as a fixed feature extractor")
    parser.add_argument("--test_only", action="store_true", help="Whether to perform inference only")

    ### Data loader
    parser.add_argument("--device", type=str, default="cpu", help="Compute architecture to use. One of ['cpu', 'cuda']")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers to run the experiment")
    parser.add_argument("--num_batches", type=int, default=430, help="Number of mini-batches to use")

    ### Data Augmentation                
    parser.add_argument("--data_resize", type=int, default=112, help="Size of each imgae sample to use")
    parser.add_argument("--data_aug_rot", type=int, default=45, help="Rotation degree for data augmentation")
    parser.add_argument("--data_aug_trans", type=float, default=0.15, help="Translation ratio for data augmentation")
    parser.add_argument("--data_aug_scale", type=float, default=0.15, help="Scale ratio for data augmentation")

    ## optimization
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate used in the scheduler")
    parser.add_argument("--lr_warmup_epochs", type=int, default=5, help="Number of epochs for learning rate warmup")
    parser.add_argument("--lr_warmup_decay", type=float, default=0.01, help="Decay ratio of learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout ratio")

    return parser


if __name__ == "__main__":
    cfg = get_args_parser().parse_args()
    wandb.init(project="chest-pathology-classification")
    wandb.run.name = wandb.run.id
    wandb.run.save()
    wandb.config.lr = 0.001
    wandb.config.update(cfg)
    main(cfg)
