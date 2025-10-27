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

def train(model, train_loader, val_loader, optimizer, res_path, device, writer, logger, start_epoch,
          rank=0, cfg=None, distri=None):
    '''Training code'''
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    logger.info("Training on rank = %d", rank)
    if rank==0: logger.info("Number of iterations in training: %d", len(train_loader))

    # Loss function
    num_iterations_per_epoch = len(train_loader)
    criterion = nn.L1Loss(reduction='none')

    # training epoch
    epochs = cfg.PG.TRAIN.EPOCHS

    loss_list = []
    best_loss = float('inf')
    epochs_since_last_improvement = 0
    best_model = None
    # scaler = torch.cuda.amp.GradScaler()

    # Load constants and teleconnection indices
    aux_constants = utils_data.loadAllConstants(device=device, cfg=cfg)  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
    upper_weights, surface_weights = aux_constants['variable_weights']

    # Train a single Pangu-Weather model
    for i in range(start_epoch, epochs + 1):
        epoch_loss = 0.0

        for id, train_data in enumerate(train_loader):
            # Load weather data at time t as the input; load weather data at time t+336 as the output
            # Note the data need to be randomly shuffled

            # Check if any of the data components are empty tensors
            if (train_data[0].sum() == 0 or
                train_data[1].sum() == 0 or
                train_data[2].sum() == 0 or
                train_data[3].sum() == 0):
                # print(f"Skipping batch {id} due to missing or empty data.")
                continue  # Skip this batch if any data component is empty

            input, input_surface, target, target_surface, periods = train_data
            input, input_surface, target, target_surface = input.to(device), input_surface.to(device), target.to(device), target_surface.to(device)

            optimizer.zero_grad()
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            # /with torch.cuda.amp.autocast():
            model.train()

            #Log scaling
            if cfg.GLOBAL.MODEL == 'All_pm':
                scale = torch.log(torch.tensor(1e20))
                input_surface[:, 4:, :, :] = ((torch.log(torch.maximum(input_surface[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)
                target_surface[:, 4:, :, :] = ((torch.log(torch.maximum(target_surface[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)

            # Note the input and target need to be normalized (done within the function)
            # Call the model and get the output
            output, output_surface = model(input,
                                           input_surface,
                                           aux_constants['weather_statistics'],
                                           aux_constants['constant_maps'],
                                           aux_constants['const_h']
                                           )  # (1,5,13,721,1440)

            # Normalize gt to make loss compariable
            target, target_surface = utils_data.normData(target, target_surface, aux_constants['weather_statistics_last'])


            ############################Surface MSE Loss###########################
            if cfg.GLOBAL.LOSS ==  "MSE":
                # We use the MAE loss to train the model
                # Different weight can be applied for different fields if needed
                loss_surface = criterion(output_surface, target_surface)

                # Cropping into a slightly larger region than MENA.
                if cfg.GLOBAL.MENA_crop:
                    # loss_surface = loss_surface[:, :, 179:388, 720:1026]
                    loss_surface = loss_surface[:, :, 175:392, 718:1030]

                weighted_surface_loss = torch.mean(loss_surface * surface_weights)

                ############################Upper MSE Loss###########################

                loss_upper = criterion(output, target)
                if cfg.GLOBAL.MENA_crop:
                    # loss_upper = loss_upper[:, :, :, 179:388, 720:1026]
                    loss_upper = loss_upper[:, :, :, 175:392, 718:1030]

                weighted_upper_loss = torch.mean(loss_upper * upper_weights)
                # The weight of surface loss is 0.25



            ############################ ExLoss ###########################
            if cfg.GLOBAL.LOSS ==  "Exloss":
                ############################Surface ExLoss###########################
                skip_batch = False

                loss_surface = []
                weighted_surface_loss = 0
                output_surface = output_surface[:, :, 175:392, 718:1030]
                target_surface = target_surface[:, :, 175:392, 718:1030]
                for j in range(surface_weights.shape[1]):
                    w = float(surface_weights[0, j, 0, 0])
                    output_surface_slice = output_surface[:, j, :, :].unsqueeze(1)    # (N, C, H, W)
                    target_surface_slice = target_surface[:, j, :, :].unsqueeze(1)  # (N, C, H, W)
                    loss = Exloss(output_surface_slice, target_surface_slice)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Skipping entire iteration due to NaN/Inf in upper channel {j}")
                        skip_batch = True
                        break  # exit the loop immediately

                    loss_surface.append(loss)
                    weighted_surface_loss += w * loss
                total_weight = surface_weights.sum(dim=1).squeeze().item()
                weighted_surface_loss /= total_weight
                # print(weighted_surface_loss.shape)
                
                ############################Upper ExLoss###########################
                loss_upper = []
                weighted_upper_loss = 0
                output = output[:, :, :, 175:392, 718:1030]
                target = target[:, :, :, 175:392, 718:1030]
                for j in range(upper_weights.shape[1]):
                    w = float(upper_weights[0, j, 0, 0, 0])
                    output_slice = output[:, j, :, :, :]    # (N, C, H, W)
                    target_slice = target[:, j, :, :, :]  # (N, C, H, W)
                    loss = Exloss(output_slice, target_slice)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Skipping entire iteration due to NaN/Inf in upper channel {j}")
                        skip_batch = True
                        break  # exit the loop immediately

                    loss_upper.append(loss)
                    weighted_upper_loss += w * loss
                total_weight = upper_weights.sum(dim=1).squeeze().item()
                weighted_upper_loss /= total_weight
                # print(weighted_upper_loss.shape)

                if skip_batch:
                    continue




            #######Total Loss######
            loss = weighted_upper_loss + weighted_surface_loss * 0.25

            # #Accounting for NaN losses.
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN or Inf at iteration {id}. Skipping this iteration.")
                continue  # Skip this batch and move to the next one

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

            if rank == 0 and id%5 == 0:
                step = num_iterations_per_epoch*(i-1) + id
                # wandb.log({"mslp_loss": torch.mean(loss_surface[0]).item()}, step=step)
                # wandb.log({"u10_loss": torch.mean(loss_surface[1]).item()}, step=step)
                # wandb.log({"v10_loss": torch.mean(loss_surface[2]).item()}, step=step)
                # wandb.log({"t2m_loss": torch.mean(loss_surface[3]).item()}, step=step)
                # wandb.log({"pm1_loss": torch.mean(loss_surface[4]).item()}, step=step)
                # wandb.log({"pm2p5_loss": torch.mean(loss_surface[5]).item()}, step=step)
                # wandb.log({"pm10_loss": torch.mean(loss_surface[6]).item()}, step=step)
                # wandb.log({"z_loss": torch.mean(loss_upper[0]).item()}, step=step)
                # wandb.log({"q_loss": torch.mean(loss_upper[1]).item()}, step=step)
                # wandb.log({"t_loss": torch.mean(loss_upper[2]).item()}, step=step)
                # wandb.log({"u_loss": torch.mean(loss_upper[3]).item()}, step=step)
                # wandb.log({"v_loss": torch.mean(loss_upper[4]).item()}, step=step)
                wandb.log({"mslp_loss": torch.mean(loss_surface[0][0]).item()}, step=step)
                wandb.log({"u10_loss": torch.mean(loss_surface[0][1]).item()}, step=step)
                wandb.log({"v10_loss": torch.mean(loss_surface[0][2]).item()}, step=step)
                wandb.log({"t2m_loss": torch.mean(loss_surface[0][3]).item()}, step=step)
                wandb.log({"pm1_loss": torch.mean(loss_surface[0][4]).item()}, step=step)
                wandb.log({"pm2p5_loss": torch.mean(loss_surface[0][5]).item()}, step=step)
                wandb.log({"pm10_loss": torch.mean(loss_surface[0][6]).item()}, step=step)
                wandb.log({"z_loss": torch.mean(loss_upper[0][0]).item()}, step=step)
                wandb.log({"q_loss": torch.mean(loss_upper[0][1]).item()}, step=step)
                wandb.log({"t_loss": torch.mean(loss_upper[0][2]).item()}, step=step)
                wandb.log({"u_loss": torch.mean(loss_upper[0][3]).item()}, step=step)
                wandb.log({"v_loss": torch.mean(loss_upper[0][4]).item()}, step=step)

                wandb.log({"train_loss": loss.item()})
                logger.info(f"Epoch {i}, Iteration {id + 1}/{len(train_loader)}: Loss = {loss.item():.6f}")
            
            torch.cuda.empty_cache()



        epoch_loss /= len(train_loader)
        if rank == 0:
            logger.info("Epoch {} : {:.3f}".format(i, epoch_loss))

        loss_list.append(epoch_loss)

        model_save_path = os.path.join(res_path, 'models')
        utils.mkdirs(model_save_path)

        # Save the training model
        if i % cfg.PG.TRAIN.SAVE_INTERVAL == 0 and rank ==0:
            if distri:
                save_file = {
                    "model": model.module.state_dict(),  # unwrap DDP
                    "optimizer": optimizer.state_dict(),
                    "epoch": i
                }
                torch.save(save_file, os.path.join(model_save_path, 'train_{}.pth'.format(i)))
                torch.save(model.module.state_dict(), os.path.join(model_save_path, 'model_weights_{}.pth'.format(i)))
                if cfg.GLOBAL.LORA:
                    peft_model = model.module if distri else model
                    if hasattr(peft_model, "save_pretrained"):
                        peft_model.save_pretrained(os.path.join(model_save_path, f"peft_model_epoch_{i}"))
                    else:
                        print("Warning: LoRA model does not support save_pretrained()")

            else:
                save_file = {"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": i}
                torch.save(save_file, os.path.join(model_save_path, 'train_{}.pth'.format(i)))
                torch.save(model.state_dict(), os.path.join(model_save_path, 'model_weights_{}.pth'.format(i)))
                if cfg.GLOBAL.LORA:
                    model.save_pretrained(os.path.join(model_save_path, f"peft_model_epoch_{i}"))

        # Begin to validate
        if i % cfg.PG.VAL.INTERVAL == 0:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0

                logger.info("Validation on rank = %d", rank)
                if rank==0: logger.info("Number of iterations in validation: %d", len(val_loader))

                for id, val_data in enumerate(val_loader, 0):

                    # Skip this batch if any data component is empty
                    if (val_data[0].sum() == 0 or 
                        val_data[1].sum() == 0 or 
                        val_data[2].sum() == 0 or 
                        val_data[3].sum() == 0
                        ):
                        # print(f"Skipping batch {id} due to missing or empty data.")
                        continue
                    
                    input_val, input_surface_val, target_val, target_surface_val, periods_val = val_data

                    if cfg.GLOBAL.MODEL == 'All_pm':
                        #Log scaling
                        scale = torch.log(torch.tensor(1e20))
                        input_surface_val[:, 4:, :, :] = ((torch.log(torch.maximum(input_surface_val[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)
                        target_surface_val[:, 4:, :, :] = ((torch.log(torch.maximum(target_surface_val[:, 4:, :, :], torch.tensor(1e-11))) - torch.log(torch.tensor(1e-11)))/ scale)


                    input_val_raw, input_surface_val_raw = input_val, input_surface_val
                    input_val, input_surface_val, target_val, target_surface_val = input_val.to(device), input_surface_val.to(device), target_val.to(device), target_surface_val.to(device)

                    # Inference
                    output_val, output_surface_val = model(input_val,input_surface_val,
                                                           aux_constants['weather_statistics'],
                                                           aux_constants['constant_maps'],
                                                           aux_constants['const_h']
                                                           )

                    # Noralize the gt to make the loss compariable
                    target_val, target_surface_val = utils_data.normData(target_val,
                                                                         target_surface_val,
                                                                         aux_constants['weather_statistics_last'])

                    val_loss_surface = criterion(output_surface_val, target_surface_val)
                    weighted_val_loss_surface = torch.mean(val_loss_surface * surface_weights)

                    val_loss_upper = criterion(output_val, target_val)
                    weighted_val_loss_upper = torch.mean(val_loss_upper * upper_weights)

                    loss = weighted_val_loss_upper + weighted_val_loss_surface * 0.25

                    val_loss += loss.item()

                    if rank == 0:
                        logger.info(f"Epoch {i}, Iteration {id + 1}/{len(val_loader)}: Loss = {loss.item():.6f}")

                if rank == 0:
                    val_loss /= len(val_loader)
                    writer.add_scalars(
                        'Loss',
                        {'train': epoch_loss,
                        'val': val_loss},
                        i
                        )
                    
                logger.info("Validate at Epoch {} : {:.3f}".format(i, val_loss))
                # Visualize the training process
                png_path = os.path.join(res_path, "png_training")
                utils.mkdirs(png_path)
                # Normalize the data back to the original space for visualization
                output_val, output_surface_val = utils_data.normBackData(output_val, output_surface_val,
                                                                            aux_constants['weather_statistics_last'])
                target_val, target_surface_val = utils_data.normBackData(target_val, target_surface_val,
                                                                            aux_constants['weather_statistics_last'])




                if cfg.GLOBAL.MODEL == 'All_pm':
                    #Log scaling
                    scale = torch.log(torch.tensor(1e20))
                    input_surface_val[:, 4:, :, :] = torch.exp(input_surface_val[:, 4:, :, :] * scale + torch.log(torch.tensor(1e-11)))
                    target_surface_val[:, 4:, :, :] = torch.exp(target_surface_val[:, 4:, :, :] * scale + torch.log(torch.tensor(1e-11)))   
                    output_surface_val[:, 4:, :, :] = torch.exp(output_surface_val[:, 4:, :, :] * scale + torch.log(torch.tensor(1e-11)))



                
                if cfg.GLOBAL.STYLE == 'output_crop' or cfg.GLOBAL.STYLE == 'padding' or cfg.GLOBAL.STYLE == 'original':
                    utils.visualize_mena(
                        output_val.detach().cpu().squeeze(),
                        target_val.detach().cpu().squeeze(),
                        input_val_raw.squeeze(),
                        var='u',
                        z=12,
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )
                
                    utils.visualize_surface_mena(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='u10',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )

                    utils.visualize_surface_mena(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='pm1',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )
                
                    utils.visualize_surface_mena(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='pm25',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )
                    
                    utils.visualize_surface_mena(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='pm10',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )
                    
                else:
                    utils.visualize_orig(
                        output_val.detach().cpu().squeeze(),
                        target_val.detach().cpu().squeeze(),
                        input_val_raw.squeeze(),
                        var='u',
                        z=12,
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )

                    utils.visuailze_surface_orig(
                        output_val.detach().cpu().squeeze(),
                        target_val.detach().cpu().squeeze(),
                        input_val_raw.squeeze(),
                        var='u',
                        z=12,
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )

                    utils.visuailze_surface_orig(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='u10',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )

                    utils.visuailze_surface_orig(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='pm2p5',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )
                    
                    utils.visuailze_surface_orig(
                        output_surface_val.detach().cpu().squeeze(),
                        target_surface_val.detach().cpu().squeeze(),
                        input_surface_val.squeeze(),
                        var='pm10',
                        step=i,
                        path=png_path,
                        cfg=cfg
                        )


                # # Early stopping
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     if not distri:
                #         torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_save_path, 'best_model.pth'))
                #     elif distri:
                #         torch.save({'model_state_dict': model.module.state_dict()}, os.path.join(model_save_path, 'best_model.pth'))
                #     logger.info(f"current best model is saved at {i} epoch.")
                #     epochs_since_last_improvement = 0
                # else:
                #     epochs_since_last_improvement += 1
                #     if epochs_since_last_improvement >= 5:
                #         logger.info(f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training.")
                #         break


                # # Early stopping
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     best_model = copy.deepcopy(model)
                #     # Save the best model
                #     torch.save(best_model, os.path.join(model_save_path, 'best_model.pth'))
                #     torch.save(best_model.state_dict(), os.path.join(model_save_path, 'best_model_nonddp.pth'))
                #     logger.info(f"current best model is saved at {i} epoch.")
                #     epochs_since_last_improvement = 0

                # else:
                #     epochs_since_last_improvement += 1
                #     if epochs_since_last_improvement >= 5:
                #         logger.info(f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training.")
                #         break


        # print("lr",lr_scheduler.get_last_lr()[0])
    return model


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
        # png_path = os.path.join(res_path, "png")
        # utils.mkdirs(png_path)

        
        # if cfg.GLOBAL.MENA_crop:
        # #['msl', 'u','v','t2m']
        #     utils.visualize(output_test.detach().cpu().squeeze(),
        #             target_test.detach().cpu().squeeze(), 
        #             input_test.detach().cpu().squeeze(),
        #             var='t',
        #             z = 2,
        #             step=target_time, 
        #             path=png_path,
        #             cfg=cfg
        #             )
            
        #     utils.visualize_surface_mena(output_surface_test.detach().cpu().squeeze(),
        #                             target_surface_test.detach().cpu().squeeze(),
        #                             input_surface_test.detach().cpu().squeeze(),
        #                             var='pm1',
        #                             step=target_time,
        #                             path=png_path,
        #                             cfg=cfg
        #                             )
            
        #     utils.visualize_surface_mena(output_surface_test.detach().cpu().squeeze(),
        #                             target_surface_test.detach().cpu().squeeze(),
        #                             input_surface_test.detach().cpu().squeeze(),
        #                             var='pm25',
        #                             step=target_time,
        #                             path=png_path,
        #                             cfg=cfg
        #                             )
            
        #     utils.visualize_surface_mena(output_surface_test.detach().cpu().squeeze(),
        #                             target_surface_test.detach().cpu().squeeze(),
        #                             input_surface_test.detach().cpu().squeeze(),
        #                             var='pm10',
        #                             step=target_time,
        #                             path=png_path,
        #                             cfg=cfg
        #                             )
            
            # utils.visualize_surface_mena(output_surface_test.detach().cpu().squeeze(),
            #                         target_surface_test.detach().cpu().squeeze(),
            #                         input_surface_test.detach().cpu().squeeze(),
            #                         var='t2m',
            #                         step=target_time,
            #                         path=png_path,
            #                         cfg=cfg
            #                         )
            
            # utils.visualize_surface_mena(output_surface_test.detach().cpu().squeeze(),
            #                         target_surface_test.detach().cpu().squeeze(),
            #                         input_surface_test.detach().cpu().squeeze(),
            #                         var='u10',
            #                         step=target_time,
            #                         path=png_path,
            #                         cfg=cfg
            #                         )

        # else:
        #     utils.visualize_orig(output_test.detach().cpu().squeeze(),
        #             target_test.detach().cpu().squeeze(), 
        #             input_test.detach().cpu().squeeze(),
        #             var='t',
        #             z = 2,
        #             step=target_time, 
        #             path=png_path,
        #             cfg=cfg
        #             )
                        
        #     utils.visuailze_surface_orig(output_surface_test.detach().cpu().squeeze(),
        #                         target_surface_test.detach().cpu().squeeze(),
        #                         input_surface_test.detach().cpu().squeeze(),
        #                         var='u10',
        #                         step=target_time,
        #                         path=png_path,
        #                         cfg=cfg
        #                         )
            
        #     utils.visuailze_surface_orig(output_surface_test.detach().cpu().squeeze(),
        #                         target_surface_test.detach().cpu().squeeze(),
        #                         input_surface_test.detach().cpu().squeeze(),
        #                         var='t2m',
        #                         step=target_time,
        #                         path=png_path,
        #                         cfg=cfg
        #                         )

        # Compute test scores
        # rmse
        output_test = output_test.squeeze()
        target_test = target_test.squeeze()
        output_surface_test = output_surface_test.squeeze()
        target_surface_test = target_surface_test.squeeze()


        rmse_upper_z[target_time] = score.weighted_rmse_torch_channels(output_test[0],
                                                                       target_test[0]).detach().cpu().numpy()
        rmse_upper_q[target_time] = score.weighted_rmse_torch_channels(output_test[1],
                                                                       target_test[1]).detach().cpu().numpy()
        rmse_upper_t[target_time] = score.weighted_rmse_torch_channels(output_test[2],
                                                                       target_test[2]).detach().cpu().numpy()
        rmse_upper_u[target_time] = score.weighted_rmse_torch_channels(output_test[3],
                                                                       target_test[3]).detach().cpu().numpy()
        rmse_upper_v[target_time] = score.weighted_rmse_torch_channels(output_test[4],
                                                                       target_test[4]).detach().cpu().numpy()

        rmse_surface[target_time] = score.weighted_rmse_torch_channels(output_surface_test,
                                                                       target_surface_test).detach().cpu().numpy()


        # acc
        surface_mean, _, upper_mean, _ = aux_constants['weather_statistics_last']
        output_test_anomaly = output_test - upper_mean.squeeze(0)
        output_surface_test_anomaly = output_surface_test - surface_mean.squeeze(0)
        target_test_anomaly = target_test - upper_mean.squeeze(0)
        target_surface_test_anomaly = target_surface_test - surface_mean.squeeze(0)

        acc_upper_z[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[0],
                                                                     target_test_anomaly[0]).detach().cpu().numpy()
        acc_upper_q[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[1],
                                                                     target_test_anomaly[1]).detach().cpu().numpy()
        acc_upper_t[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[2],
                                                                     target_test_anomaly[2]).detach().cpu().numpy()
        acc_upper_u[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[3],
                                                                     target_test_anomaly[3]).detach().cpu().numpy()
        acc_upper_v[target_time] = score.weighted_acc_torch_channels(output_test_anomaly[4],
                                                                     target_test_anomaly[4]).detach().cpu().numpy()

        acc_surface[target_time] = score.weighted_acc_torch_channels(output_surface_test_anomaly,
                                                                     target_surface_test_anomaly).detach().cpu().numpy()
    # Save rmses to csv
    csv_path = os.path.join(res_path, "csv5_world")
    utils.mkdirs(csv_path)
    utils.save_errorScores(csv_path, rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v, rmse_surface, "rmse", cfg=cfg)
    utils.save_errorScores(csv_path, acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v, acc_surface, "acc", cfg=cfg)


if __name__ == "__main__":
    pass
