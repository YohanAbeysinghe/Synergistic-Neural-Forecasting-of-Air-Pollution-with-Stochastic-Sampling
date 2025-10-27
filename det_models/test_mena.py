import sys
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")

import os
import copy
import torch
from torch import nn
import wandb
from era5_data import score
from era5_data import utils, utils_data
from loss.exloss import Exloss
from torch.cuda.amp import autocast, GradScaler


def test(test_loader, model, device, res_path, cfg):
    # set up empty dics for rmses and anormaly correlation coefficients

    rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v = dict(), dict(), dict(), dict(), dict()
    rmse_surface = dict()

    acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v = dict(), dict(), dict(), dict(), dict()
    acc_surface = dict()

    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)

    batch_id = 0
    for id, data in enumerate(test_loader, 0):
        
        # Check if any of the data components are empty tensors
        if (data[0].sum() == 0 or
            data[1].sum() == 0 or
            data[2].sum() == 0 or
            data[3].sum() == 0):
            # print(f"Skipping batch {id} due to missing or empty data.")
            continue  # Skip this batch if any data component is empty

        # Store initial input for different models
        print(f"predict on {id}")
        input_test, input_surface_test, target_test, target_surface_test, periods_test = data
        input_test, input_surface_test, target_test, target_surface_test = input_test.to(device), input_surface_test.to(device), target_test.to(device), target_surface_test.to(device)
        model.eval()


        if cfg.GLOBAL.MODEL == 'All_pm':
            #Log scaling
            scale = torch.log(torch.tensor(1e20))
            input_surface_test[:, 4:, :, :] = ((torch.log(torch.maximum(input_surface_test[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)
            target_surface_test[:, 4:, :, :] = ((torch.log(torch.maximum(target_surface_test[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)


        # Inference
        output_test, output_surface_test = model(input_test, input_surface_test,
                                                 aux_constants['weather_statistics'],
                                                 aux_constants['constant_maps'], aux_constants['const_h'])
        # Transfer to the output to the original data range
        output_test, output_surface_test = utils_data.normBackData(output_test, output_surface_test,
                                                        aux_constants['weather_statistics_last'])
        

        if cfg.GLOBAL.MODEL == 'All_pm':
            #Log scaling
            scale = torch.log(torch.tensor(1e20))
            input_surface_test[:, 4:, :, :] = torch.exp(input_surface_test[:, 4:, :, :] * scale + torch.log(torch.tensor(1e-11)))
            target_surface_test[:, 4:, :, :] = torch.exp(target_surface_test[:, 4:, :, :] * scale + torch.log(torch.tensor(1e-11)))   
            output_surface_test[:, 4:, :, :] = torch.exp(output_surface_test[:, 4:, :, :] * scale + torch.log(torch.tensor(1e-11)))


        target_time = periods_test[1][batch_id]

        # Visualize
        png_path = os.path.join(res_path, "png5_")
        utils.mkdirs(png_path)


        input_test = input_test[:, :, :, 175:392, 718:1030]
        output_test =output_test[:, :, :, 175:392, 718:1030]
        target_test = target_test[:, :, :, 175:392, 718:1030]
        output_surface_test = output_surface_test[:, :, 175:392, 718:1030]
        target_surface_test = target_surface_test[:, :, 175:392, 718:1030]
        input_surface_test = input_surface_test[:, :, 175:392, 718:1030]

        

                
        # if id % 1 == 0:
        #     utils.visualize_mena(
        #         output_test.detach().cpu().squeeze(),
        #         target_test.detach().cpu().squeeze(),
        #         input_test.squeeze(),
        #         var='u',
        #         z=12,
        #         step=target_time,
        #         path=png_path,
        #         cfg=cfg
        #         )
        
        #     utils.visualize_surface_mena(
        #         output_surface_test.detach().cpu().squeeze(),
        #         target_surface_test.detach().cpu().squeeze(),
        #         input_surface_test.squeeze(),
        #         var='u10',
        #         step=target_time,
        #         path=png_path,
        #         cfg=cfg
        #         )

        #     utils.visualize_surface_mena(
        #         output_surface_test.detach().cpu().squeeze(),
        #         target_surface_test.detach().cpu().squeeze(),
        #         input_surface_test.squeeze(),
        #         var='pm1',
        #         step=target_time,
        #         path=png_path,
        #         cfg=cfg
        #         )
        
        #     utils.visualize_surface_mena(
        #         output_surface_test.detach().cpu().squeeze(),
        #         target_surface_test.detach().cpu().squeeze(),
        #         input_surface_test.squeeze(),
        #         var='pm25',
        #         step=target_time,
        #         path=png_path,
        #         cfg=cfg
        #         )
            
        #     utils.visualize_surface_mena(
        #         output_surface_test.detach().cpu().squeeze(),
        #         target_surface_test.detach().cpu().squeeze(),
        #         input_surface_test.squeeze(),
        #         var='pm10',
        #         step=target_time,
        #         path=png_path,
        #         cfg=cfg
        #         )
            


        # Compute test scores
        # rmse
        output_test = output_test.squeeze()
        target_test = target_test.squeeze()
        output_surface_test = output_surface_test.squeeze()
        target_surface_test = target_surface_test.squeeze()


        rmse_upper_z[target_time] = score.weighted_rmse_cropped(output_test[0],
                                                                       target_test[0]).detach().cpu().numpy()
        rmse_upper_q[target_time] = score.weighted_rmse_cropped(output_test[1],
                                                                       target_test[1]).detach().cpu().numpy()
        rmse_upper_t[target_time] = score.weighted_rmse_cropped(output_test[2],
                                                                       target_test[2]).detach().cpu().numpy()
        rmse_upper_u[target_time] = score.weighted_rmse_cropped(output_test[3],
                                                                       target_test[3]).detach().cpu().numpy()
        rmse_upper_v[target_time] = score.weighted_rmse_cropped(output_test[4],
                                                                       target_test[4]).detach().cpu().numpy()

        rmse_surface[target_time] = score.weighted_rmse_cropped(output_surface_test,
                                                                       target_surface_test).detach().cpu().numpy()


        # acc
        surface_mean, _, upper_mean, _ = aux_constants['weather_statistics_last']
        output_test_anomaly = output_test - upper_mean.squeeze(0)
        output_surface_test_anomaly = output_surface_test - surface_mean.squeeze(0)
        target_test_anomaly = target_test - upper_mean.squeeze(0)
        target_surface_test_anomaly = target_surface_test - surface_mean.squeeze(0)

        acc_upper_z[target_time] = score.weighted_acc_cropped(output_test_anomaly[0],
                                                                     target_test_anomaly[0]).detach().cpu().numpy()
        acc_upper_q[target_time] = score.weighted_acc_cropped(output_test_anomaly[1],
                                                                     target_test_anomaly[1]).detach().cpu().numpy()
        acc_upper_t[target_time] = score.weighted_acc_cropped(output_test_anomaly[2],
                                                                     target_test_anomaly[2]).detach().cpu().numpy()
        acc_upper_u[target_time] = score.weighted_acc_cropped(output_test_anomaly[3],
                                                                     target_test_anomaly[3]).detach().cpu().numpy()
        acc_upper_v[target_time] = score.weighted_acc_cropped(output_test_anomaly[4],
                                                                     target_test_anomaly[4]).detach().cpu().numpy()

        acc_surface[target_time] = score.weighted_acc_cropped(output_surface_test_anomaly,
                                                                     target_surface_test_anomaly).detach().cpu().numpy()
    # Save rmses to csv
    csv_path = os.path.join(res_path, "csv5_")
    utils.mkdirs(csv_path)
    utils.save_errorScores(csv_path, rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_surface, "rmse", cfg=cfg)
    utils.save_errorScores(csv_path, acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v, acc_surface, "acc", cfg=cfg)


if __name__ == "__main__":
    pass