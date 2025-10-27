import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys

sys.path.append("/home/yohan.abeysinghe/pangu-pytorch")
from typing import Tuple, List
import torch
import random
from torch.utils import data
from torchvision import transforms as T
import os
import time
from torch.nn.modules.module import _addindent
import matplotlib.pyplot as plt
import logging


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass





def visualize_mena(output, target, input, var, z, step, path, cfg):
    variables = cfg.ERA5_UPPER_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(20, 2))
    
    # Ensure all inputs are NumPy arrays
    # Compute vmin and vmax across all relevant data for consistent color scaling
    output = output.detach().cpu().numpy() if not isinstance(output, np.ndarray) else output
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # outputcrop = output[var, z, 179:388, 720:1026]
    # targetcrop = target[var, z, 179:388, 720:1026]
    # inputcrop = input[var, z, 179:388, 720:1026]

    # vmax = max(outputcrop.max(), targetcrop.max(), inputcrop.max())
    # vmin = min(outputcrop.min(), targetcrop.min(), inputcrop.min())
    vmax = max(target[var, z].max(), input[var, z].max())
    vmin = min(target[var, z].min(), input[var, z].min())

    ax1 = fig.add_subplot(151)
    plot1 = ax1.imshow(input[var, z, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('input')

    # New subplot for the sliced region of 'input'
    ax2 = fig.add_subplot(152)
    plot2 = ax2.imshow(input[var, z, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('input_slice')

    ax3 = fig.add_subplot(153)
    plot3 = ax3.imshow(target[var, z, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('gt')

    ax4 = fig.add_subplot(154)
    plot4 = ax4.imshow(output[var, z, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('pred')

    ax5 = fig.add_subplot(155)
    plot5 = ax5.imshow(output[var, z, :, :] - target[var, z, :, :], cmap="RdBu_r")
    plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
    ax5.title.set_text('bias')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}_Z{}'.format(step, variables[var], z)))
    plt.close(fig)


def visualize_surface_mena(output, target, input, var, step, path, cfg):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(20, 2))

    # Ensure all inputs are NumPy arrays
    # Compute vmin and vmax across all relevant data for consistent color scaling
    output = output.detach().cpu().numpy() if not isinstance(output, np.ndarray) else output
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # outputcrop = output[var, 179:388, 720:1026]
    # targetcrop = target[var, 179:388, 720:1026]
    # inputcrop = input[var, 179:388, 720:1026]
    # combined = np.concatenate([outputcrop.flatten(),targetcrop.flatten(),inputcrop.flatten()])

    # Use percentiles for robust color scaling, which ignores extreme outliers
    if var==4 or var==5 or var==6:
        combined = np.concatenate([input[var].flatten(), target[var].flatten()])
        vmin = np.percentile(combined, 5)
        vmax = np.percentile(combined, 95)
    else:
        vmax = max(target[var].max(), input[var].max())
        vmin = min(target[var].min(), input[var].min())

    # vmax = max(output.max(), target.max(), input.max())
    # vmin = min(output.min(), target.min(), input.min())


    ax1 = fig.add_subplot(151)
    plot1 = ax1.imshow(input[var, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot1 = ax1.imshow(input[var, :, :], cmap="RdBu")
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('input')

    # New subplot for the sliced region of 'input'
    ax2 = fig.add_subplot(152)
    plot2 = ax2.imshow(input[var, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot2 = ax2.imshow(input[var, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('input_slice')

    ax3 = fig.add_subplot(153)
    plot3 = ax3.imshow(target[var, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot3 = ax3.imshow(target[var, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('gt')

    ax4 = fig.add_subplot(154)
    plot4 = ax4.imshow(output[var, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot4 = ax4.imshow(output[var, :, :], cmap="RdBu")
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('pred')

    ax5 = fig.add_subplot(155)
    plot5 = ax5.imshow(output[var, :, :] - target[var, :, :], cmap="RdBu_r")
    plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
    ax5.title.set_text('bias')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
    plt.close(fig)






# def visualize_surface(output, target, input, var, step, path, cfg):
#     variables = cfg.ERA5_SURFACE_VARIABLES
#     var = variables.index(var)
#     fig = plt.figure(figsize=(20, 2))

#     # Ensure all inputs are NumPy arrays
#     # Compute vmin and vmax across all relevant data for consistent color scaling
#     output = output.detach().cpu().numpy() if not isinstance(output, np.ndarray) else output
#     target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
#     input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input


#     vmax = max(output.max(), target.max(), input.max())
#     vmin = min(output.min(), target.min(), input.min())

#     # Use percentiles for robust color scaling, which ignores extreme outliers
#     # if var==4 or var==5 or var==6:
#     #     vmin = np.percentile(input[var, :, :], 0)
#     #     vmax = np.percentile(input[var, :, :], 80)
#     # else:
#     #     vmin = input[var, :, :].min()
#     #     vmax = input[var, :, :].max()

#     ax1 = fig.add_subplot(151)
#     plot1 = ax1.imshow(input[var, :, :], cmap="RdBu", vmin=vmin, vmax=vmax)
#     plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
#     ax1.title.set_text('input')


#     ax3 = fig.add_subplot(152)
#     plot3 = ax3.imshow(target[var, :, :], cmap="RdBu", vmin=vmin, vmax=vmax)
#     plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
#     ax3.title.set_text('gt')

#     ax4 = fig.add_subplot(153)
#     plot4 = ax4.imshow(output[var, :, :], cmap="RdBu", vmin=vmin, vmax=vmax)
#     plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
#     ax4.title.set_text('pred')

#     ax5 = fig.add_subplot(154)
#     plot5 = ax5.imshow(output[var, :, :] - target[var, :, :], cmap="RdBu")
#     plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
#     ax5.title.set_text('bias')

#     plt.tight_layout()
#     plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
#     plt.close(fig)


def visuailze_surface_orig(output, target, input, var, step, path, cfg):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))

    # Convert all tensors to NumPy
    input_np = input[var, :, :].detach().cpu().numpy()
    target_np = target[var, :, :].detach().cpu().numpy()
    output_np = output[var, :, :].detach().cpu().numpy()
    bias_np = output_np - target_np

    ax3 = fig.add_subplot(141)
    plot3 = ax3.imshow(input_np, cmap="RdBu")
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('input')

    ax2 = fig.add_subplot(142)
    plot2 = ax2.imshow(target_np, cmap="RdBu")
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('gt')

    ax1 = fig.add_subplot(143)
    plot1 = ax1.imshow(output_np, cmap="RdBu")
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('pred')

    ax4 = fig.add_subplot(144)
    plot4 = ax4.imshow(bias_np, cmap="RdBu")
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('bias')

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(fname=os.path.join(path, '{}_{}.png'.format(step, variables[var])))
    plt.close(fig)

def visualize_orig(output, target, input, var, z, step, path, cfg):
    variables = cfg.ERA5_UPPER_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))

    ax3 = fig.add_subplot(141)
    plot3 = ax3.imshow(input[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('input')

    ax2 = fig.add_subplot(142)
    plot2 = ax2.imshow(target[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('gt')

    ax1 = fig.add_subplot(143)
    plot1 = ax1.imshow(output[var, z, :, :], cmap="RdBu")  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('pred')

    ax4 = fig.add_subplot(144)
    plot4 = ax4.imshow(output[var, z, :, :] - target[var, z, :, :], cmap="RdBu")
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('bias')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}_Z{}'.format(step, variables[var], z)))
    plt.close(fig)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def torch_summarize(model, show_weights=False, show_parameters=False, show_gradients=False):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    tmpstr += ', total parameters={}'.format(total_params)
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, module.parameters())])
        weights = tuple([tuple(p.size()) for p in filter(lambda p: p.requires_grad, module.parameters())])
        grads = tuple([str(p.requires_grad) for p in filter(lambda p: p.requires_grad, module.parameters())])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        if show_gradients:
            tmpstr += ', gradients={}'.format(grads)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def save_errorScores(csv_path, z, q, t, u, v, surface, error, cfg):
    score_upper_z = pd.DataFrame.from_dict(z,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_q = pd.DataFrame.from_dict(q,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_t = pd.DataFrame.from_dict(t,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_u = pd.DataFrame.from_dict(u,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_upper_v = pd.DataFrame.from_dict(v,
                                orient='index',
                                columns=cfg.ERA5_UPPER_LEVELS)
    score_surface = pd.DataFrame.from_dict(surface,
                                orient='index',
                                columns=cfg.ERA5_SURFACE_VARIABLES)

    score_upper_z.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_z'))
    score_upper_q.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_q'))
    score_upper_t.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_t'))
    score_upper_u.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_u'))
    score_upper_v.to_csv("{}/{}.csv".format(csv_path, f'{error}_upper_v'))
    score_surface.to_csv("{}/{}.csv".format(csv_path, f'{error}_surface'))

if __name__ == "__main__":

    """
    """


