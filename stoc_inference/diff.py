import os
import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import utils
from final_plotting import visualize_surface_gulf_cartopy, visualize_surface_MENA_cartopy
import torch
from era5_data import utils_data
from utils import load_model_g, diffusion_inverse_transform, merge_pred, normalize_numpy, merge_pred_test
from normalization import normBack


def mkdirs(path):
    os.makedirs(path, exist_ok=True)

def diff(test_loader, stoch_model, deter_model, device, cfg):

    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)

    climat = torch.load('/l/users/fahad.khan/akhtar/Pangu/ExtremeCast/data/running_average.pt') # [69, 721, 1440]
    climat = climat[-3:, 175:392, 718:1030] # [3, 217, 312]
    climat = climat.unsqueeze(0) # [1, 3, 217, 312]

    batch_id = 0

    for id, test_data in enumerate(test_loader):

        if (test_data[0].sum() == 0 or test_data[1].sum() == 0 or test_data[2].sum() == 0 or test_data[3].sum() == 0):
            continue

        print(f"predict on {id}")

        input, input_surface, target, target_surface, periods = test_data
        input, input_surface, target, target_surface = input.to(device), input_surface.to(device), target.to(device), target_surface.to(device)
        target_, target_surface_ = target.clone(), target_surface.clone()


        deter_model.eval()
        stoch_model.eval()
        stoch_model = stoch_model.to(device)

        if cfg.GLOBAL.MODEL == 'All_pm':
            #Log scaling
            scale = torch.log(torch.tensor(1e20))
            input_surface[:, 4:, :, :] = ((torch.log(torch.maximum(input_surface[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)
            target_surface_[:, 4:, :, :] = ((torch.log(torch.maximum(target_surface_[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)


        # Inference
        output_test, output_surface_test = deter_model(input, input_surface,
                                                aux_constants['weather_statistics'],
                                                aux_constants['constant_maps'],
                                                aux_constants['const_h'])

        deterministic_prediction = output_surface_test[:, 4:, :, :].clone()

        climat1, climat2 = utils_data.normData(target_, target_surface_, aux_constants['weather_statistics_last']) 

        # Condition should be only 69 channels.
        output_test = output_test.reshape(1, -1, 721, 1440)
        condition = torch.cat([output_test, output_surface_test[:, 3:, :, :]], dim=1)
        condition = condition[:, :, 175:392, 718:1030]
        condition = condition.float().to(device) # [1, 69, 217, 312]

        with torch.no_grad():
            model_output = stoch_model.sample(condition = condition)
        diffusion_out = diffusion_inverse_transform(model_output)

        stochastic_prediction = merge_pred_test(diffusion_out, deterministic_prediction[:, :, 175:392, 718:1030], climat, climat2)

        input_pm = input_surface[:, 4:, :, :]
        deterministic_prediction_pm = normBack(deterministic_prediction, aux_constants['weather_statistics_last'])
        stochastic_prediction_pm = normBack(stochastic_prediction, aux_constants['weather_statistics_last'])

        if cfg.GLOBAL.MODEL == 'All_pm':
            #Log scaling
            scale = torch.log(torch.tensor(1e20))
            deterministic_prediction_pm = torch.exp(deterministic_prediction_pm * scale + torch.log(torch.tensor(1e-11)))
            stochastic_prediction_pm = torch.exp(stochastic_prediction_pm * scale + torch.log(torch.tensor(1e-11)))
            input_pm = torch.exp(input_pm * scale + torch.log(torch.tensor(1e-11)))


        target_time = periods[1][batch_id]

        # Visualize
        res_path = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/extreamcast_results/aurora_comparison"
        png_path = os.path.join(res_path, "comparison_4")
        mkdirs(png_path)


        visualize_surface_gulf_cartopy(
            deterministic_prediction_pm[:, :, 175:392, 718:1030].detach().cpu().squeeze(),
            stochastic_prediction_pm.detach().cpu().squeeze(),
            target_surface[:, 4:, 175:392, 718:1030].detach().cpu().squeeze(),
            input_pm[:, :, 175:392, 718:1030].squeeze(),
            var='pm1',
            step=target_time,
            path=png_path,
            cfg=cfg
            )

        visualize_surface_gulf_cartopy(
            deterministic_prediction_pm[:, :, 175:392, 718:1030].detach().cpu().squeeze(),
            stochastic_prediction_pm.detach().cpu().squeeze(),
            target_surface[:, 4:, 175:392, 718:1030].detach().cpu().squeeze(),
            input_pm[:, :, 175:392, 718:1030].squeeze(),
            var='pm25',
            step=target_time,
            path=png_path,
            cfg=cfg
            )
    
        visualize_surface_gulf_cartopy(
            deterministic_prediction_pm[:, :, 175:392, 718:1030].detach().cpu().squeeze(),
            stochastic_prediction_pm.detach().cpu().squeeze(),
            target_surface[:, 4:, 175:392, 718:1030].detach().cpu().squeeze(),
            input_pm[:, :, 175:392, 718:1030].squeeze(),
            var='pm10',
            step=target_time,
            path=png_path,
            cfg=cfg
            )


        # visualize_surface_MENA_cartopy(
        #     deterministic_prediction_pm[:, :, 175:392, 718:1030].detach().cpu().squeeze(),
        #     stochastic_prediction_pm.detach().cpu().squeeze(),
        #     target_surface[:, 4:, 175:392, 718:1030].detach().cpu().squeeze(),
        #     input_pm[:, :, 175:392, 718:1030].squeeze(),
        #     var='pm1',
        #     step=target_time,
        #     path=png_path,
        #     cfg=cfg
        #     )

        # visualize_surface_MENA_cartopy(
        #     deterministic_prediction_pm[:, :, 175:392, 718:1030].detach().cpu().squeeze(),
        #     stochastic_prediction_pm.detach().cpu().squeeze(),
        #     target_surface[:, 4:, 175:392, 718:1030].detach().cpu().squeeze(),
        #     input_pm[:, :, 175:392, 718:1030].squeeze(),
        #     var='pm25',
        #     step=target_time,
        #     path=png_path,
        #     cfg=cfg
        #     )
    
        # visualize_surface_MENA_cartopy(
        #     deterministic_prediction_pm[:, :, 175:392, 718:1030].detach().cpu().squeeze(),
        #     stochastic_prediction_pm.detach().cpu().squeeze(),
        #     target_surface[:, 4:, 175:392, 718:1030].detach().cpu().squeeze(),
        #     input_pm[:, :, 175:392, 718:1030].squeeze(),
        #     var='pm10',
        #     step=target_time,
        #     path=png_path,
        #     cfg=cfg
        #     )
 