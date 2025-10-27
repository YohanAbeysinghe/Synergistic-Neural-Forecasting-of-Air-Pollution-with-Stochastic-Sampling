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
from models.test_mena import test
# from models.pangu_sample import test

import os
import wandb
import copy
import logging
import argparse
import importlib
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter


def load_model_for_inference(cfg, output_path, device):
    # Step 1: Reconstruct base model
    model = PanguModel(device=device, cfg=cfg).to(device)

    # Step 2: Edit conv_surface weights for input/output layers
    checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device)
    state_dict = checkpoint['model']

    if cfg.GLOBAL.MODEL == 'All_pm':
        # Learning rate for new variables.
        model_state_dict = model.state_dict()

        # Modify input layer for dimension matching and loading the existing weights
        # for first 112 channels. Rest is initialized randomly.
        new_input_weight = torch.zeros((192, 160, 1))
        new_input_weight[:, :112, :] = state_dict['_input_layer.conv_surface.weight']
        nn.init.xavier_uniform_(new_input_weight[:, 112:, :])
        state_dict['_input_layer.conv_surface.weight'] = new_input_weight

        # Modify output layer for dimension matching and loading the existing weights
        # for first 64 channels. Rest is initialized randomly.
        new_output_weight = torch.zeros((112, 384, 1))
        new_output_weight[:64, :, :] = state_dict['_output_layer.conv_surface.weight']
        nn.init.xavier_uniform_(new_output_weight[64:, :, :])
        state_dict['_output_layer.conv_surface.weight'] = new_output_weight

        # Modify output layer bias. Loading first 64 biases.
        new_output_bias = torch.zeros(112)
        new_output_bias[:64] = state_dict['_output_layer.conv_surface.bias']
        state_dict['_output_layer.conv_surface.bias'] = new_output_bias


    model.load_state_dict(state_dict, strict=False)

    # Step 3: Apply LoRA
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name)
            print(f"Appended module for LoRA: {name}")


    lora_config = LoraConfig(
        r=cfg.PG.TRAIN.LOW_RANK,          # Make sure this is capitalized consistently
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        # bias="none",                      # or "all" / "lora_only" depending on needs
        # task_type="REGRESSION"           # Or "FEATURE_EXTRACTION" if only embedding
    )

    model = get_peft_model(model, lora_config).to(device)

    # Step 4: Load LoRA and edited-layer weights from finetuned checkpoint
    best_model_path = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/results/lora_full_finetune_loss_cropped/models/model_weights_3.pth"
    # model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Step 4: Load edited-layer weights from finetuned checkpoint

    # best_model_path = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/results/Full_finetune_normalized_0602_1/models/model_weights_1.pth"

    # best_model_path = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/results/proper_crop_mena/models/model_weights_1.pth"
    state_dict = torch.load(best_model_path, map_location=device)


    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]  # Remove 'module.' prefix
        else:
            new_k = k
        new_state_dict[new_k] = v


    # Load the modified state dict
    model.load_state_dict(new_state_dict)



    model.eval()
    return model



###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='config8', help='Option to load different configs')
parser.add_argument('--output', type=str, default='lora_full_finetune_loss_cropped', help='Name of the output directory')
parser.add_argument('--distri', default=False, help='Doing the distributed training')
args = parser.parse_args()

config_module = importlib.import_module(f"configs.{args.config}")
cfg = config_module.cfg
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
best_model = load_model_for_inference(cfg, output_path, device)


test(test_loader=test_dataloader,
     model=best_model,
     device=device,
     res_path=output_path,
     cfg = cfg)
#