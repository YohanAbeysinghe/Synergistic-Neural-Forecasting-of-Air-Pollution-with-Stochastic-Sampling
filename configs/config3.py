import sys
from .ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 1 # @Yohan
__C.GLOBAL.SEED =99
__C.GLOBAL.NUM_THREADS = 2
__C.GLOBAL.MODEL = 'original'
__C.GLOBAL.MENA_crop = False #False
__C.GLOBAL.LOSS = "MSE" #"Exloss"
__C.GLOBAL.START = "checkpoint" #"scratch"
__C.GLOBAL.LORA = False #False

for dirs in ['/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch']:
    if os.path.exists(dirs):
        __C.GLOBAL.PATH = dirs
assert __C.GLOBAL.PATH is not None

__C.PG_INPUT_PATH = '/l/users/fahad.khan/akhtar/Pangu/data/pangu_data'
assert __C.PG_INPUT_PATH is not None

__C.PG_OUT_PATH = os.path.join('/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/results')
assert __C.PG_OUT_PATH is not None

__C.ERA5_UPPER_LEVELS = ['1000','925','850', '700','600','500','400', '300','250', '200','150','100', '50']
__C.ERA5_SURFACE_VARIABLES = ['msl','u10','v10','t2m', 'pm1', 'pm25', 'pm10']
__C.ERA5_UPPER_VARIABLES = ['z','q','t','u','v']

__C.PG = edict()
__C.PG.HORIZON = 24
__C.PG.TRAIN = edict()
__C.PG.TRAIN.EPOCHS = 5
__C.PG.TRAIN.LR = 5e-4 #5e-6 #5e-4
__C.PG.TRAIN.WEIGHT_DECAY = 3e-7 #3e-6
__C.PG.TRAIN.START_TIME =  '20040101' #'20030101'
__C.PG.TRAIN.END_TIME = '20040106'
__C.PG.TRAIN.FREQUENCY = '12h'
__C.PG.TRAIN.BATCH_SIZE = 1
__C.PG.TRAIN.UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
__C.PG.TRAIN.SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00, 1.20, 1.20, 1.20]
__C.PG.TRAIN.SAVE_INTERVAL = 1
__C.PG.TRAIN.LOW_RANK = 8


__C.PG.VAL = edict()
__C.PG.VAL.START_TIME = '20050101'
__C.PG.VAL.END_TIME = '20050106'
__C.PG.VAL.FREQUENCY = '12h'
__C.PG.VAL.BATCH_SIZE = 1
__C.PG.VAL.INTERVAL = 1


__C.PG.TEST = edict()
__C.PG.TEST.START_TIME = '20090101'
__C.PG.TEST.END_TIME = '20090106'
__C.PG.TEST.FREQUENCY = '12h'
__C.PG.TEST.BATCH_SIZE = 1

__C.PG.BENCHMARK = edict()
__C.PG.BENCHMARK.PRETRAIN_24 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_24.onnx')
__C.PG.BENCHMARK.PRETRAIN_6 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_6.onnx')
__C.PG.BENCHMARK.PRETRAIN_3 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_3.onnx')
__C.PG.BENCHMARK.PRETRAIN_1 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_1.onnx')
__C.PG.BENCHMARK.PRETRAIN_24_fp16 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model_fp16/pangu_weather_24_fp16.onnx')
__C.PG.BENCHMARK.PRETRAIN_24_torch = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_24_torch.pth')
  
__C.MODEL = edict()



# __C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))