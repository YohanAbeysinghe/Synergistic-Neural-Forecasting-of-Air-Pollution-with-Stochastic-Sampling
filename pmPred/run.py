import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")
import torch
from diff import diff
import torch.nn as nn
from torch.utils import data
from era5_data import utils_data
from load_model import load_model_for_inference
from config import cfg

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
best_model = load_model_for_inference(cfg, device)


diff(test_loader=test_dataloader, model=best_model, device=device, cfg = cfg)