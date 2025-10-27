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
from models.pangu_sample import test, train

import os
import wandb
import copy
import logging
import argparse
import importlib
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter

###########################################################################################
############################# Argument Parsing ############################################
###########################################################################################
#
parser = argparse.ArgumentParser(description="Pangu Model Training")
parser.add_argument('--config', type=str, default='config9', help='Option to load different configs')
parser.add_argument('--output', type=str, default='lora_full_finetune_final', help='Name of the output directory')
parser.add_argument('--distri', default=True, help='Doing the distributed training')
args = parser.parse_args()

config_module = importlib.import_module(f"configs.{args.config}")
cfg = config_module.cfg
#
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(cfg.GLOBAL.NUM_THREADS)
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
output_path = os.path.join(cfg.PG_OUT_PATH, args.output)
utils.mkdirs(output_path)

writer_path = os.path.join(output_path, "writer")
if not os.path.exists(writer_path):
    os.mkdir(writer_path)
writer = SummaryWriter(writer_path)

logger_name = "finetune_fully" + str(cfg.PG.HORIZON)
utils.logger_info(logger_name, os.path.join(output_path, logger_name + '.log'))
logger = logging.getLogger(logger_name)
#
###########################################################################################
############################## Distributed Training #######################################
###########################################################################################
#
if not args.distri:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    local_rank=0
    num_gpus = 1

if args.distri:
    def setup_distributed():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank

    def cleanup_distributed():
        dist.destroy_process_group()

    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Using device: {device}")

    num_gpus = torch.cuda.device_count()
    if local_rank == 0:
        logger.info(f"Number of GPUs available: {num_gpus}")
#
###########################################################################################
################################### Data Loading ##########################################
###########################################################################################
#
PATH = cfg.PG_INPUT_PATH

train_dataset = utils_data.NetCDFDataset(
    nc_path=PATH,
    data_transform=None,
    training=True,
    validation = False,
    startDate = cfg.PG.TRAIN.START_TIME,
    endDate= cfg.PG.TRAIN.END_TIME,
    freq=cfg.PG.TRAIN.FREQUENCY,
    horizon=cfg.PG.HORIZON,
    cfg=cfg
    )

if args.distri:
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True
        )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.PG.TRAIN.BATCH_SIZE//num_gpus,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler
        )

else:
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.PG.TRAIN.BATCH_SIZE,
        drop_last=True,
        shuffle=True,
        num_workers=0,
        pin_memory=False
        )


val_dataset = utils_data.NetCDFDataset(
    nc_path=PATH,
    data_transform=None,
    training=False,
    validation = True,
    startDate = cfg.PG.VAL.START_TIME,
    endDate= cfg.PG.VAL.END_TIME,
    freq=cfg.PG.VAL.FREQUENCY,
    horizon=cfg.PG.HORIZON,
    cfg=cfg
    )

if args.distri:
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=True,
        drop_last=True
        )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.PG.VAL.BATCH_SIZE//num_gpus,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler
        )

else:
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.PG.VAL.BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=0,
        pin_memory=False
        )

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
#
###########################################################################################
################################## WandB ##################################################
###########################################################################################
#
# Initialize W&B with your project name and hyperparameters
os.environ["WANDB_API_KEY"] = "f26dcc1314b4959cd257db827dcdcff1a2e54f2e"

if local_rank == 0:
    wandb.init(project="climate_modeling", name=args.output, config={
        "learning_rate": cfg.PG.TRAIN.LR,
        "batch_size": cfg.PG.TRAIN.BATCH_SIZE,
        "num_epochs": cfg.PG.TRAIN.EPOCHS,
        "num_gpus": num_gpus,
        "start_time": cfg.PG.TRAIN.START_TIME,
        "end_time": cfg.PG.TRAIN.END_TIME,
        "output_path": output_path,
        # "pm2.5_weightage": cfg.PG.TRAIN.SURFACE_WEIGHTS[4],
        "Law_rank": cfg.PG.TRAIN.LOW_RANK,
        # "Model": cfg.PG.GLOBAL.MODEL,
        # "Loss": cfg.GLOBAL.LOSS,
        # "start": cfg.GLOBAL.START
    })
#
###########################################################################################
###################################Loading Checkpoint######################################
###########################################################################################
#
model = PanguModel(device=device, cfg=cfg).to(device)

module_copy = copy.deepcopy(model) # For later comparisons


if cfg.GLOBAL.START == "checkpoint":
  checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=False, map_location='cuda')
  state_dict = checkpoint['model']
  #
  ###########################################################################################
  ################### Editing Checkpoint to Get New Variables ###############################
  ###########################################################################################
  #
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

  # Load the modified state_dict if cfg.GLOBAL.MODEL == 'pm25'.
  model.load_state_dict(state_dict, strict=False)
  #
###########################################################################################
####################################Hyperparameters########################################
###########################################################################################
#
# Dynamically find all nn.Linear layers in the model
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

#Unfreezing changed layers
# Ensure manually modified layers are set to trainable
model._input_layer.conv_surface.weight.requires_grad = True
model._output_layer.conv_surface.weight.requires_grad = True
model._output_layer.conv_surface.bias.requires_grad = True

# # Ensure cropped earth_specific_bias tensors are trainable
# for layer in [0, 1, 2, 3]:
#     for block in range(6):
#         try:
#             attn = eval(f"model.layers.EarthSpecificLayer{layer}.blocks.EarthSpecificBlock{block}.attention")
#             if hasattr(attn, 'earth_specific_bias') and isinstance(attn.earth_specific_bias, torch.nn.Parameter):
#                 attn.earth_specific_bias.requires_grad = True
#         except:
#             pass

# # Print trainable parameters (first few lines only)
# print("\n[LoRA parameters only]")
# for name, param in model.named_parameters():
#     if "lora_" in name and param.requires_grad:
#         print(f"✓ {name}")

# print("\n[Original model parameters being trained]")
# for name, param in model.named_parameters():
#     if "lora_" not in name and param.requires_grad:
#         print(f"✓ {name}")

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
# original_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad)

# print(f"\nParameter Breakdown:")
# print(f"  LoRA Parameters           : {lora_params:,}")
# print(f"  Original Trainable Params : {original_params:,}")
# print(f"  Total Trainable Params    : {trainable_params:,}")

# print(f"\nModel Size:")
# print(f"  Total Parameters          : {total_params:,}")
# print(f"  Trainable Parameters      : {trainable_params:,}")
# print(f"  Percentage Trainable      : {100 * trainable_params / total_params:.2f}%")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr = cfg.PG.TRAIN.LR,
                             weight_decay= cfg.PG.TRAIN.WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[25, 50],
                                                    gamma=0.5)

start_epoch = 1
#
###########################################################################################
############################## Logging Info ###############################################
###########################################################################################
#
msg = '\n'
msg += utils.torch_summarize(model, show_weights=False)
logger.info(msg)

print("weather statistics are loaded!")
#
###########################################################################################
############################## Train and Validation #######################################
###########################################################################################
#
if args.distri:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if local_rank == 0:
    wandb.watch(model, log="all", log_freq=100)

model = train(model,
              train_loader=train_dataloader,
              val_loader=val_dataloader,
              optimizer=optimizer,
              # lr_scheduler=lr_scheduler,
              res_path = output_path,
              device=device,
              writer=writer, 
              logger = logger,
              start_epoch=start_epoch,
              cfg = cfg,
              rank=local_rank,
              distri=args.distri)

# peft_model = train(
#         peft_model,
#         train_loader=train_dataloader,
#         val_loader=val_dataloader,
#         optimizer=optimizer,
#         # lr_scheduler=lr_scheduler,
#         res_path = output_path,
#         device=device,
#         writer=writer, 
#         logger = logger,
#         start_epoch=start_epoch,
#         cfg = cfg,
#         rank=local_rank
#         )
#
###########################################################################################
################################### Testing  ##############################################
###########################################################################################
#
# best_model = torch.load(os.path.join(output_path,"models/best_model.pth"),
#                         map_location='cuda:0',
#                         weights_only=False)

# logger.info("Begin testing...")

# test(test_loader=test_dataloader,
#      model=best_model,
#      device=device,
#      res_path=output_path,
#      cfg = cfg)
#
###########################################################################################
###########################################################################################