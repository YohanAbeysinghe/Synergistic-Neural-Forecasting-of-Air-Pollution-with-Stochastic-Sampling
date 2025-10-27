
import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")

import os
import torch
import wandb
import torch.nn as nn
from config import cfg
from torch.utils import data
from diff import diff
from utils import load_model_g, load_pm_trained_diff
from era5_data import utils_data
from load_model import load_model_for_inference

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
local_rank=0
num_gpus = 1

PATH = cfg.PG_INPUT_PATH

test_dataset = utils_data.NetCDFDataset(
    nc_path=PATH,
    data_transform=None,
    training=False,
    validation=False,
    startDate=cfg.PG.TEST.START_TIME,
    endDate=cfg.PG.TEST.END_TIME,
    freq=cfg.PG.TEST.FREQUENCY,
    horizon=cfg.PG.HORIZON,
    cfg=cfg
    )

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    batch_size=cfg.PG.TEST.BATCH_SIZE,
    drop_last=True,
    shuffle=False,
    num_workers=0,
    pin_memory=False
    )

# Load model for inference
stoch_model = load_pm_trained_diff(diffision_path='/l/users/fahad.khan/akhtar/Pangu/ExtremeCast/checkpoints/model_g.pth')
best_model = load_model_for_inference(cfg, device)


diff(test_loader=test_dataloader, stoch_model=stoch_model, deter_model=best_model, device=device, cfg = cfg)