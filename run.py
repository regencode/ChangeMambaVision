import argparse
import os
import yaml
from types import SimpleNamespace
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from ChangeMambaVision.utils.train_test_val_lightning import ChangeDetectionModel
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import ChangeMambaVision
from ChangeMambaVision.datasets.levir_cd import LEVIR_CD_Dataset, load_levir
from ChangeMambaVision.utils.train_test_val import train_one_epoch, test_one_epoch
from ChangeMambaVision.utils.display import display_images
import mamba_ssm
from ChangeMambaVision.models.CDMamba.models.CDMamba import CDMamba
from ChangeMambaVision.utils import augmentations as A
import glob, shutil, os
from pyunpack import Archive
from torch.utils.data import DataLoader
from torchinfo import summary

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def main(config):
    SELECTED_DATASET = config.experiment.dataset_name
    SEED = config.experiment.seed

    MODEL_NAME = f"{config.experiment.model_name}_seed{SEED}_{SELECTED_DATASET}" # for file naming

    SAVE_FOLDER_PATH = config.paths.save_folder
    DATA_SOURCE = config.paths.data_source
    ROOT = config.paths.root


    EPOCHS = config.training.epochs
    TRAIN_BATCH_SIZE = config.training.batch_size.train
    VAL_BATCH_SIZE = config.training.batch_size.val
    TEST_BATCH_SIZE = config.training.batch_size.test
    # SGD
    OPTIM_KWARGS = {
        "optim": config.optimizer.type,
        "lr": config.optimizer.lr,
        "momentum": config.optimizer.momentum,
        "weight_decay": config.optimizer.weight_decay
    }
    SCHEDULER_KWARGS = {
        "start_factor": config.scheduler.start_factor,
        "end_factor": config.scheduler.end_factor,
        "total_iters": EPOCHS # match number of epochs
    }

    train_transforms = A.PairCompose([
        A.RandomHorizontalFlip(p=0.5),
        A.RandomVerticalFlip(p=0.5),
        A.RandomResizedCropPair(
            size=(256, 256),
            scale=(0.8, 1.0),
            ratio=(1/1, 1/1)
        ),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch.device: {device}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED) # for the current GPU
    torch.cuda.manual_seed_all(SEED) # for all GPUs
    pl.seed_everything(SEED, workers=True)

    ##############################################################

    MODEL_CONFIG = vars(config.model)
    norm = vars(MODEL_CONFIG["norm"])
    print(norm)
    MODEL_CONFIG["norm"] = [norm["type"], {"num_groups": vars(norm["num_groups"])}]
    model = CDMamba(**MODEL_CONFIG)
    model = model.to("cuda")
    summary(model, input_size=((1, 3, 256, 256), (1, 3, 256, 256)))

    # Load dataset if its not loaded yet, and patchify
    load_levir(DATA_SOURCE, patchify=True, patch_size=(256, 256))
    loss_fn = torch.nn.CrossEntropyLoss()
    logger = CSVLogger(
        save_dir=os.path.join(SAVE_FOLDER_PATH, "logs"),
        name=f"{MODEL_NAME}-logs",
    )

    ckpt_dir = os.path.join(SAVE_FOLDER_PATH, f"{MODEL_NAME}-checkpoints")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",        # must match self.log key
        mode="min",                # lower val_loss is better
        save_top_k=1,              # keep only the best model
        save_last=True,            # also save last epoch
        filename="best-{epoch:03d}-{val_loss:.4f}",
    )

    lit_model = ChangeDetectionModel(
        model=model,
        loss_fn=loss_fn,
        optim_kwargs=OPTIM_KWARGS,
        scheduler_kwargs=SCHEDULER_KWARGS
    )

    if SELECTED_DATASET == "levir":
        train_data = LEVIR_CD_Dataset(split="train", pair_transforms=train_transforms)
        val_data = LEVIR_CD_Dataset(split="val")
        test_data = LEVIR_CD_Dataset(split="test")
    elif SELECTED_DATASET == "whu":
        train_data = LEVIR_CD_Dataset(split="train", pair_transforms=train_transforms)
        val_data = LEVIR_CD_Dataset(split="val")
        test_data = LEVIR_CD_Dataset(split="test")
    else:
        print("no valid dataset selected")
        return

    # Reduced num_workers to 2 to prevent shared memory crashes
    train_loader = DataLoader(
        train_data,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    if not config.experiment.is_train:
        # if not training...
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=0,   # skip training loop
            limit_val_batches=1,     # MUST run validation once
            enable_checkpointing=True,
            callbacks=[checkpoint_cb]
        )
        ckpt_last = os.path.join(ckpt_dir, "last.ckpt")
        # single val batch in order to load "best" checkpoint
        try:
            trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_last if os.path.exists(ckpt_last) else None)
        except:
            pass
    else:
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",   # "auto" also works
            devices=1,
            precision="bf16-mixed",
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            logger=logger,
            log_every_n_steps=10,
            callbacks=[checkpoint_cb],
        )

        ckpt_last = os.path.join(ckpt_dir, "last.ckpt")
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_last if os.path.exists(ckpt_last) else None)

    # Test model after training
    trainer.test(
        lit_model,
        dataloaders=test_loader,
        ckpt_path="best"
    )
    print(torch.cuda.max_memory_allocated() / 1024**2, "MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    print(config_dict)
    config = dict_to_namespace(config_dict)
    main(config)
