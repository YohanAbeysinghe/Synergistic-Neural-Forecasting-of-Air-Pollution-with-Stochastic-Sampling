import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")

import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda.amp import autocast
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# from era5_data.config import cfg
from era5_data import utils_data, utils
from models.pangu_model import PanguModel
# from models.pangu_sample import test, train
from models.test_mena import test

import os
import wandb
import copy
import logging
import argparse
import importlib
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='f', help='Option to load different configs')
parser.add_argument('--output', type=str, default='Original_Pangu_Inference_MENA_721', help='Name of the output directory')
parser.add_argument('--distri', default=False, help='Doing the distributed training')
args = parser.parse_args()

config_module = importlib.import_module(f"configs.{args.config}")
cfg = config_module.cfg
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
local_rank=0
num_gpus = 1

output_path = os.path.join(cfg.PG_OUT_PATH, args.output)
#
###########################################################################################
###########################################################################################
###########################################################################################
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
model = PanguModel(device=device, cfg=cfg).to(device)

# Step 2: Edit conv_surface weights for input/output layers
checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device)
state_dict = checkpoint['model']

model.load_state_dict(state_dict)

model.eval()


# Testing on the Whole Globe
test(test_loader=test_dataloader,
     model=model,
     device=device,
     res_path=output_path,
     cfg = cfg)
#



