import torch
import torch.nn as nn
from models.pangu_model import PanguModel
from peft import LoraConfig, get_peft_model


def load_model_for_inference(cfg, device):
    # Step 1: Reconstruct base model
    model = PanguModel(device=device, cfg=cfg).to(device)

    # Step 2: Edit conv_surface weights for input/output layers
    checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device)
    state_dict = checkpoint['model']

    if cfg.GLOBAL.MODEL == 'All_pm':
        # Learning rate for new variables.
        model_state_dict = model.state_dict()

        new_input_weight = torch.zeros((192, 160, 1))
        new_input_weight[:, :112, :] = state_dict['_input_layer.conv_surface.weight']
        nn.init.xavier_uniform_(new_input_weight[:, 112:, :])
        state_dict['_input_layer.conv_surface.weight'] = new_input_weight

        new_output_weight = torch.zeros((112, 384, 1))
        new_output_weight[:64, :, :] = state_dict['_output_layer.conv_surface.weight']
        nn.init.xavier_uniform_(new_output_weight[64:, :, :])
        state_dict['_output_layer.conv_surface.weight'] = new_output_weight

        # Modify output layer bias. Loading first 64 biases.
        new_output_bias = torch.zeros(112)
        new_output_bias[:64] = state_dict['_output_layer.conv_surface.bias']
        state_dict['_output_layer.conv_surface.bias'] = new_output_bias

    # Cropping the bias
    # This ensures the earth_specific_bias from the checkpoint matches the current model's expected size (e.g., [1, 32, 6, 144, 144])
    # for layer in [0, 3]:
    #     for block in [0, 1]:
    #         key = f'layers.EarthSpecificLayer{layer}.blocks.EarthSpecificBlock{block}.attention.earth_specific_bias'
    #         if key in state_dict:
    #             orig_bias = state_dict[key]
    #             if orig_bias.shape[1] > 40:
    #                 state_dict[key] = orig_bias[:, 29:69, :, :, :]

    # # Crop earth_specific_bias from checkpoint to match current model size
    # for layer in [1, 2]:  # EarthSpecificLayer1 and 2
    #     for block in range(6):  # Each has 6 EarthSpecificBlocks
    #         key = f'layers.EarthSpecificLayer{layer}.blocks.EarthSpecificBlock{block}.attention.earth_specific_bias'
    #         if key in state_dict:
    #             orig_bias = state_dict[key]
    #             target_shape = model.state_dict()[key].shape
    #             if orig_bias.shape != target_shape:
    #                 print(f'Cropping {key} from {orig_bias.shape} to {target_shape}')
    #                 state_dict[key] = orig_bias[:, 15:35, :, :, :]

    model.load_state_dict(state_dict)

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
        lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)

    # Step 4: Load LoRA and edited-layer weights from finetuned checkpoint
    best_model_path = "/l/users/fahad.khan/akhtar/Pangu/data/pangu_data/results/lora_full_finetune/models/model_weights_5.pth"
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