import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append("/l/users/fahad.khan/akhtar/Pangu/pangu-pytorch")

import torch
import wandb
import numpy as np
from era5_data import utils_data
from normalization import normBack
from utils import load_model_g, diffusion_inverse_transform, merge_pred, normalize_numpy

save_dir = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/extreamcast_results/test1/"

def train(test_loader, stoch_model, deter_model, device, cfg, epochs=1):

    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)

    start_epoch = 0
    print(len(test_loader), 'iteration in test_loader')

    for i in range(start_epoch, epochs + 1):

        epoch_loss = 0.0

        for id, train_data in enumerate(test_loader):

            if (train_data[0].sum() == 0 or train_data[1].sum() == 0 or train_data[2].sum() == 0 or train_data[3].sum() == 0):
                continue

            input, input_surface, target, target_surface, periods = train_data
            input, input_surface, target, target_surface = input.to(device), input_surface.to(device), target.to(device), target_surface.to(device)

            # print(f"predict on {id}")

            deter_model.eval()
            stoch_model.train()
            stoch_model = stoch_model.to(device)

            optimizer = torch.optim.Adam(stoch_model.parameters(), lr=cfg.PG.TRAIN.LR)



            if cfg.GLOBAL.MODEL == 'All_pm':
                #Log scaling
                scale = torch.log(torch.tensor(1e20))
                input_surface[:, 4:, :, :] = ((torch.log(torch.maximum(input_surface[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)
                target_surface[:, 4:, :, :] = ((torch.log(torch.maximum(target_surface[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)

            # Inference
            output_test, output_surface_test = deter_model(input, input_surface,
                                                    aux_constants['weather_statistics'],
                                                    aux_constants['constant_maps'],
                                                    aux_constants['const_h'])

            target, target_surface = utils_data.normData(target, target_surface, aux_constants['weather_statistics_last'])


            # Condition should be only 69 channels.
            output_test = output_test.reshape(1, -1, 721, 1440)
            condition = torch.cat([output_test, output_surface_test[:, 3:, :, :]], dim=1)
            condition = condition[:, :, 175:392, 718:1030]
            condition = condition.float().to(device) # [1, 69, 217, 312]

            # Extract diffusion input and conditioning
            # original_target = target_surface_test2[:, 4:, :, :]
            original_target = target_surface[:, 4:, 175:392, 718:1030]
            original_target = original_target.float().to(device) #[1, 3, 217, 312]

            # Give a mask to focus on the MENA area.
            mask =  mask = torch.ones(217, 312, device=device) #[217, 312]
            # mask =  mask = torch.ones(721, 1440, device=device)

            loss = stoch_model(img=original_target, condition=condition, mask=mask) 

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {i}, Iteration {id}, Loss: {loss.item():.4f}")
            wandb.log({"epoch": i, "iteration": id, "loss": loss.item()})

            epoch_loss += loss.item()

            if id % 300 == 0:
                checkpoint_dir = os.path.join(save_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "checkpoints", f"stoch_model_epoch_{i}_batch_{id}.pth")
                torch.save(stoch_model.state_dict(), save_path)
                print(f"Saved model to {save_path}")

        epoch_loss /= len(test_loader)
        print(f"Epoch {i} completed. Average Loss: {epoch_loss:.4f}")
        wandb.log({"epoch_avg_loss": epoch_loss, "epoch": i})                   
