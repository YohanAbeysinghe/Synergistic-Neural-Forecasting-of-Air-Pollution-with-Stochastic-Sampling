import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")

import os
import torch
import wandb
import torch.nn as nn
from config import cfg
from torch.utils import data
from diff_sample import train
from utils import load_model_g
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

# os.environ["WANDB_API_KEY"] = "f26dcc1314b4959cd257db827dcdcff1a2e54f2e"
# wandb.init(project="diffusion-training", config=cfg, name="test2")

# Load model for inference
stoch_model = load_model_g()
best_model = load_model_for_inference(cfg, device)

train(test_loader=test_dataloader, stoch_model=stoch_model, deter_model=best_model, device=device, cfg=cfg, epochs=5)