import torch
import torch.nn as nn
from torch.functional import F
import numpy as np
from model.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from collections import OrderedDict
import onnx
import onnxruntime as ort
from peft import get_peft_model, LoraConfig, TaskType

DATA_DIR = "/l/users/fahad.khan/akhtar/Pangu/ExtremeCast/data"

# data_mean = np.load(f"{DATA_DIR}/data_mean.npy")[np.newaxis, :, np.newaxis, np.newaxis]
# data_std = np.load(f"{DATA_DIR}/data_std.npy")[np.newaxis, :, np.newaxis, np.newaxis]
data_mean = np.load(f"{DATA_DIR}/true_mean.npy")[np.newaxis, :, np.newaxis, np.newaxis]
data_std = np.load(f"{DATA_DIR}/true_std.npy")[np.newaxis, :, np.newaxis, np.newaxis]

diffusion_min = np.load(f"{DATA_DIR}/diffusion_min.npy")[np.newaxis, :, np.newaxis, np.newaxis]
diffusion_max = np.load(f"{DATA_DIR}/diffusion_max.npy")[np.newaxis, :, np.newaxis, np.newaxis]

min_logvar = np.load(f"{DATA_DIR}/min_logvar.npy")
max_logvar = np.load(f"{DATA_DIR}/max_logvar.npy")


def normalize_numpy(data):
    return (data-data_mean)/data_std

def normalize_torch(data):
    data_mean_ = torch.tensor(data_mean, device=data.device)
    data_std_ = torch.tensor(data_std, device=data.device)
    return (data-data_mean_)/data_std_

def inverse_normalize_numpy(data):
    return data*data_std+data_mean

def inverse_normalize_torch(data):
    data_mean_ = torch.tensor(data_mean, device=data.device)
    data_std_ = torch.tensor(data_std, device=data.device)
    return data*data_std_+data_mean_

def diffusion_inverse_transform(diffusion_out):
    '''
    Min-max normalization is used during training in diffusion, 
    turning it into Z-Score normalization.
    '''
    _, C, _, _ =  diffusion_out.shape
    data_min= torch.tensor(diffusion_min[:,:C], device=diffusion_out.device)
    data_max = torch.tensor(diffusion_max[:,:C], device=diffusion_out.device)
    return diffusion_out * (data_max - data_min) + data_min


def load_model_d(path='./checkpoints/model_d.onnx'):
    print("[It takes about a few minutes]")
    model = onnx.load(path)
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(path, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
    print("[Model_d load completed]")
    return ort_session


def load_model_g(diffision_path='/l/users/fahad.khan/akhtar/Pangu/ExtremeCast/checkpoints/model_g.pth'):
    u_net_model = Unet(
                    dim = 128,
                    init_dim = None,
                    out_dim = 3,
                    dim_mults = (1, 2, 4, 8),
                    channels = 69+3,
                    self_condition = False,
                    resnet_block_groups = 4,
                    learned_variance = False,
                    learned_sinusoidal_cond = False,
                    random_fourier_features = False,
                    learned_sinusoidal_dim = 16,
                    sinusoidal_pos_emb_theta = 10000,
                    attn_dim_head = 32,
                    attn_heads = 4,
                    full_attn = None,
                    flash_attn = True)

    model = GaussianDiffusion(
                model=u_net_model,
                image_size = [721, 1440],
                timesteps = 1000,
                sampling_timesteps = 20,
                objective = 'pred_noise',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                ddim_sampling_eta = 0.99,
                auto_normalize = True,
                offset_noise_strength = 0.,
                min_snr_loss_weight = False,
                min_snr_gamma = 5
            )

    checkpoint_dict = torch.load(diffision_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint_dict['model']
    for key in checkpoint_model:
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model[key].items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.model.eval()
    model.eval()
    print("[Model_g load completed]")
    return model


def load_model_g_with_lora(diffision_path='/l/users/fahad.khan/akhtar/Pangu/ExtremeCast/checkpoints/model_g.pth',
                            lora_r=16,
                            lora_alpha=16,
                            lora_dropout=0.1,
                            device=torch.device('cuda')
                            ):
    u_net_model = Unet(
                    dim = 128,
                    init_dim = None,
                    out_dim = 3,
                    dim_mults = (1, 2, 4, 8),
                    channels = 69+3,
                    self_condition = False,
                    resnet_block_groups = 4,
                    learned_variance = False,
                    learned_sinusoidal_cond = False,
                    random_fourier_features = False,
                    learned_sinusoidal_dim = 16,
                    sinusoidal_pos_emb_theta = 10000,
                    attn_dim_head = 32,
                    attn_heads = 4,
                    full_attn = None,
                    flash_attn = True)

    model = GaussianDiffusion(
                model=u_net_model,
                image_size = [721, 1440],
                timesteps = 1000,
                sampling_timesteps = 20,
                objective = 'pred_noise',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                ddim_sampling_eta = 0.99,
                auto_normalize = True,
                offset_noise_strength = 0.,
                min_snr_loss_weight = False,
                min_snr_gamma = 5
            )

    checkpoint_dict = torch.load(diffision_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint_dict['model']
    for key in checkpoint_model:
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model[key].items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)

    # # Dynamically find all Linear layers
    # target_modules = []
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Linear):
    #         target_modules.append(name)
    #         print(f"Appended module for LoRA: {name}")

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['linear']  # PEFT will now find all nn.Linear layers automatically
)


    model.model = get_peft_model(model.model, peft_config)
    model.to(device)
    model.eval()

    print("[Model_g with LoRA loaded]")
    return model


def load_pm_trained_diff(diffision_path='/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/extreamcast_results/test1/checkpoints/stoch_model_epoch_2_batch_9900.pth'):
    u_net_model = Unet(
                    dim = 128,
                    init_dim = None,
                    out_dim = 3,
                    dim_mults = (1, 2, 4, 8),
                    channels = 69+3,
                    self_condition = False,
                    resnet_block_groups = 4,
                    learned_variance = False,
                    learned_sinusoidal_cond = False,
                    random_fourier_features = False,
                    learned_sinusoidal_dim = 16,
                    sinusoidal_pos_emb_theta = 10000,
                    attn_dim_head = 32,
                    attn_heads = 4,
                    full_attn = None,
                    flash_attn = True)

    model = GaussianDiffusion(
                model=u_net_model,
                image_size = [721, 1440],
                timesteps = 1000,
                sampling_timesteps = 20,
                objective = 'pred_noise',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                ddim_sampling_eta = 0.99,
                auto_normalize = True,
                offset_noise_strength = 0.,
                min_snr_loss_weight = False,
                min_snr_gamma = 5
            )
    
    # # Step 2: Load the flat state_dict
    # checkpoint_model = torch.load(diffision_path, map_location='cpu')

    # # Step 3: Clean up keys if needed (e.g., remove DDP prefix)
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint_model.items():
    #     name = k[len("module."):] if k.startswith("module.") else k
    #     new_state_dict[name] = v

    # # Step 4: Load the state_dict
    # model.load_state_dict(new_state_dict, strict=True)
    # model.eval()
    # print("[Model_g loaded from state_dict]")
    # return model
    
    # Step 2: Load checkpoint
    checkpoint = torch.load(diffision_path, map_location='cpu')

    # Step 3: Extract only the model weights
    checkpoint_model = checkpoint["model"]

    # Step 4: Clean up keys if needed (remove 'module.' from DDP)
    new_state_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        name = k[len("module."):] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Step 5: Load into your model
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    print("[Model_g loaded from state_dict]")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


def adjust_mean(tensor, target_tensor):
    '''
    Adjust the output of diffusion so that the output of diffusion has the same mean as the output of model_d. 
    This can eliminate the bias between the two models.
    '''
    m1, m2, m3 = tensor[:, 0].mean(), tensor[:, 1].mean(), tensor[:, 2].mean()
    m4, m5, m6 = target_tensor[:, 0].mean(), target_tensor[:, 1].mean(), target_tensor[:, 2].mean()

    tensor[:, 0] -= m1
    tensor[:, 1] -= m2
    tensor[:, 2] -= m3

    tensor[:, 0] += m4
    tensor[:, 1] += m5
    tensor[:, 2] += m6

    return tensor

def get_target_mask(tar, tar_clim, th=0.1):
    '''
    Based on the model predictions and the climatology at that moment, the masks of extremely large and extremely small are obtained.
    Mask_up reflects areas with significantly higher than average weather in the past. 
    For example, for temperature, mask_up refers to areas with abnormally high temperatures.
    '''
    tar_diff = tar-tar_clim
    
    n, c, h, w = tar.shape
    up_th = 1-th
    down_th = th
    tar_up =  torch.quantile(tar_diff.view(n, c, h*w), q=up_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # 4, 69, 1, 1
    tar_down =  torch.quantile(tar_diff.view(n, c, h*w), q=down_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # 4, 69, 1, 1

    mask_up = ((tar_diff - tar_up) > 0).float()
    mask_down = ((tar_diff - tar_down) < 0).float()

    mask = ((mask_up + mask_down) > 0).float()

    return mask, mask_up, mask_down


# def merge_pred(diffusion_pred, pred, climat, d_use_k=0.1):
#     '''
#     Use mask to merge the output of the diffusion model and the output of the deterministic model. 
#     Specifically, for the mask_up and mask_down regions, we use the output of the diffusion model, 
#     and for other regions, we use the output of the deterministic model. 
#     Such a merging method can give full play to the accuracy of the predictions of the deterministic model 
#     and the extremeness of the predictions of the generation model.
#     '''
#     _, C, _, _ =  diffusion_pred.shape
#     mask, mask_up, mask_down = get_target_mask(pred[:,:C], climat[:,:C])

#     mask_up = mask_up * 0.4
#     mask_down = mask_down * 0.4

#     diffusion_pred = adjust_mean(diffusion_pred, pred[:,:C])

#     bigger = ((diffusion_pred-pred[:,:C]) > 0).float()
#     use_diffusion_bigger = ((bigger + mask_up) > 1.9).float()
#     smaller = ((diffusion_pred-pred[:,:C]) < 0).float()
#     use_diffusion_smaller = ((smaller + mask_down) > 1.9).float()
#     use_diffusion = ((use_diffusion_bigger+use_diffusion_smaller) > 0).float()*d_use_k
#     use_pred = 1.0 - use_diffusion

#     pred_merge = use_pred*pred[:,:C] + use_diffusion*diffusion_pred
    
#     return torch.cat([pred_merge, pred[:,C:]],dim=1)

def merge_pred(diffusion_pred, pred, climat, climat2, d_use_k=0.3, top_k=2000):
    '''
    Blend deterministic prediction and climatology at the top K and bottom K points per sample.
    '''
    B, C, H, W = pred.shape
    climat2_cropped = climat2[:, 4:, 175:392, 718:1030]

    final_merge = pred[:, :C].clone()

    for b in range(B):
        # Flatten per sample
        flat_climat = climat2_cropped[b].reshape(-1)

        # Get top K indices
        topk_values, topk_indices = torch.topk(flat_climat, top_k)

        # Get bottom K indices
        bottomk_values, bottomk_indices = torch.topk(-flat_climat, top_k)

        # Create per-sample mask
        mask = torch.zeros_like(flat_climat)
        mask[topk_indices] = 1
        mask[bottomk_indices] = 1
        mask = mask.view_as(climat2_cropped[b])

        # Blend per sample
        blended = d_use_k * pred[b, :C] + (1 - d_use_k) * climat2_cropped[b]
        final_merge[b] = torch.where(mask == 1, blended, pred[b, :C])

    return torch.cat([final_merge, pred[:, C:]], dim=1)


# def merge_pred_test(diffusion_pred, pred, climat, climat2, d_use_k=0.5):
#     '''
#     Blend deterministic prediction and climatology only on the top 20% of the highest value points in pred.
#     '''
#     _, C, H, W = pred.shape

#     # Crop climat2 to match the shape
#     climat2_cropped = climat2[:, 4:, 175:392, 718:1030]

#     final_merge = pred[:, :C].clone()

#     for b in range(pred.shape[0]):
#         # Flatten per sample
#         flat_pred = pred[b, :C].reshape(-1)

#         # Find the value at the 80th percentile (top 20% threshold)
#         threshold = torch.quantile(flat_pred, 0.80)

#         # Create mask for top 20% points
#         mask = (pred[b, :C] >= threshold).float()

#         # Blend only at masked locations
#         blended = d_use_k * pred[b, :C] + (1 - d_use_k) * climat2_cropped[b]
#         final_merge[b] = torch.where(mask == 1, blended, pred[b, :C])

#     return torch.cat([final_merge, pred[:, C:]], dim=1)

def merge_pred_test(diffusion_pred, pred, climat, climat2, d_use_k=0.2):
    '''
    Blend deterministic prediction and climatology only on the top 20% and bottom 20% of the points in pred.
    '''
    _, C, H, W = pred.shape

    # Crop climat2 to match the shape
    climat2_cropped = climat2[:, 4:, 175:392, 718:1030]

    final_merge = pred[:, :C].clone()

    for b in range(pred.shape[0]):
        # Flatten per sample
        flat_pred = pred[b, :C].reshape(-1)

        # Calculate top and bottom thresholds
        top_threshold = torch.quantile(flat_pred, 0.80)
        bottom_threshold = torch.quantile(flat_pred, 0.00)

        # Create mask for top 20% or bottom 20% points
        mask = ((pred[b, :C] >= top_threshold) | (pred[b, :C] <= bottom_threshold)).float()

        # Blend only at masked locations
        blended = d_use_k * pred[b, :C] + (1 - d_use_k) * climat2_cropped[b]
        final_merge[b] = torch.where(mask == 1, blended, pred[b, :C])

    return torch.cat([final_merge, pred[:, C:]], dim=1)


def get_scale(pred, div_rate=3.0):
    '''
    The scale parameter used in the ExBooster module can be hyperparameters or learnable parameters. 
    We use "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" to learn scale.
    '''
    max_logvar_ = torch.tensor(max_logvar, device=pred.device)
    min_logvar_ = torch.tensor(min_logvar, device=pred.device)

    pred_std = pred[:,69:]
    B, C, H, W = pred_std.shape
    pred_std = pred_std.reshape(B, -1)
    pred_std = max_logvar_ - F.softplus(max_logvar_ - pred_std)
    pred_std = min_logvar_ + F.softplus(pred_std - min_logvar_)

    pred_std = pred_std.reshape(B, C, H, W) / 2
    pred_std = torch.exp(pred_std) / div_rate
    return pred_std