import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

# def visualize_surface_gulf_cartopy(output_deter, output_stoch, target, input, var, step, path, cfg):

#     output_deter = output_deter[:, 25:-72, 122:-70]
#     output_stoch = output_stoch[:, 25:-72, 122:-70]
#     target = target[:, 25:-72, 122:-70]
#     input = input[:, 25:-72, 122:-70]

#     variables = cfg.ERA5_SURFACE_VARIABLES
#     var = variables.index(var)
#     fig = plt.figure(figsize=(16, 3))

#     # Ensure all inputs are NumPy arrays
#     # Compute vmin and vmax across all relevant data for consistent color scaling
#     output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
#     output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
#     target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
#     input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

#     # Use percentiles for robust color scaling, which ignores extreme outliers
#     combined = np.concatenate([input[var-4].flatten(), target[var-4].flatten(), output_deter[var-4].flatten(), output_stoch[var-4].flatten()])
#     vmin = np.percentile(combined, 0)
#     vmax = np.percentile(combined, 99)
#     data_range = vmax - vmin

#     # Extend the range by 5% on both sides
#     vmin = vmin + 0.15 * data_range
#     vmax = vmax - 0.05 * data_range


#     proj = ccrs.PlateCarree()
#     extent = [30, 60, 10, 40]


#     ax1 = fig.add_subplot(141, projection=proj)
#     ax1.set_extent(extent, crs=proj)
#     plot1 = ax1.imshow(input[var-4, :, :], cmap="Blues", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
#     ax1.coastlines(resolution='10m', color='black', linewidth=1)
#     ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
#     # plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
#     ax1.title.set_text('input')

#     ax2 = fig.add_subplot(142, projection=proj)
#     ax2.set_extent(extent, crs=proj)
#     plot2 = ax2.imshow(output_deter[var-4, :, :], cmap="Blues", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
#     ax2.coastlines(resolution='10m', color='black', linewidth=1)
#     ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
#     # plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
#     ax2.title.set_text('output_deter')

#     ax3 = fig.add_subplot(143, projection=proj)
#     ax3.set_extent(extent, crs=proj)
#     plot3 = ax3.imshow(output_stoch[var-4, :, :], cmap="Blues", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
#     ax3.coastlines(resolution='10m', color='black', linewidth=1)
#     ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
#     # plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
#     ax3.title.set_text('output_stoch')

#     ax4 = fig.add_subplot(144, projection=proj)
#     ax4.set_extent(extent, crs=proj)
#     plot4 = ax4.imshow(target[var-4, :, :], cmap="Blues", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
#     ax4.coastlines(resolution='10m', color='black', linewidth=1)
#     ax4.add_feature(cfeature.BORDERS, linewidth=0.5)
#     plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
#     ax4.title.set_text('target')

#     bias_deter = output_deter[var-4, :, :] - target[var-4, :, :]
#     bias_stoch = output_stoch[var-4, :, :] - target[var-4, :, :]

#     all_bias = np.concatenate([bias_deter.flatten(), bias_stoch.flatten()])
#     bias_limit = np.percentile(np.abs(all_bias), 100) + np.percentile(np.abs(all_bias), 10)  # Get 95th percentile of absolute bias

#     bias_min = -bias_limit
#     bias_max = bias_limit

#     # bias_max = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 0)
#     # bias_min = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 95)

#     # ax5 = fig.add_subplot(165, projection=proj)
#     # ax5.set_extent(extent, crs=proj)
#     # plot5 = ax5.imshow(bias_deter, cmap="RdBu", vmin=bias_min, vmax=bias_max, extent=extent, transform=proj, origin='upper')
#     # ax5.coastlines(resolution='10m', color='black', linewidth=1)
#     # ax5.add_feature(cfeature.BORDERS, linewidth=0.5)
#     # # plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
#     # ax5.title.set_text('bias_output_deter')

#     # ax6 = fig.add_subplot(166, projection=proj)
#     # ax6.set_extent(extent, crs=proj)
#     # plot6 = ax6.imshow(bias_stoch, cmap="RdBu", vmin=bias_min, vmax=bias_max, extent=extent, transform=proj, origin='upper')
#     # ax6.coastlines(resolution='10m', color='black', linewidth=1)
#     # ax6.add_feature(cfeature.BORDERS, linewidth=0.5)
#     # # plt.colorbar(plot6, ax=ax6, fraction=0.05, pad=0.05)
#     # ax6.title.set_text('bias_output_stoch')

#     plt.tight_layout()
#     plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
#     plt.close(fig)


# def visualize_surface_gulf_cartopy(output_deter, output_stoch, target, input, var, step, path, cfg):

#     # Crop to region
#     output_deter = output_deter[:, 25:-72, 122:-70]
#     output_stoch = output_stoch[:, 25:-72, 122:-70]
#     target = target[:, 25:-72, 122:-70]
#     input = input[:, 25:-72, 122:-70]

#     variables = cfg.ERA5_SURFACE_VARIABLES
#     var = variables.index(var)

#     # Prepare figure
#     fig = plt.figure(figsize=(20, 6))

#     # Ensure numpy arrays
#     output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
#     output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
#     target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
#     input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

#     # Compute color scale
#     combined = np.concatenate([input[var-4].flatten(), target[var-4].flatten(), output_deter[var-4].flatten(), output_stoch[var-4].flatten()])
#     vmin = np.percentile(combined, 0)
#     vmax = np.percentile(combined, 99)
#     data_range = vmax - vmin
#     vmin = vmin + 0.15 * data_range
#     vmax = vmax - 0.05 * data_range

#     vmin= 1.0e-09
#     vmax= 75.0e-09

#     proj = ccrs.PlateCarree()
#     extent = [30, 60, 10, 40]

#     # Plot titles and data
#     titles = ['Input', 'Output (Deterministic)', 'Output (Stochastic)', 'Target']
#     data = [input[var-4], output_deter[var-4], output_stoch[var-4], target[var-4]]

#     # Convert step string to datetime
#     dt = datetime.strptime(step, "%Y%m%d%H")

#     for idx in range(4):
#         ax = fig.add_subplot(1, 4, idx + 1, projection=proj)
#         ax.set_extent(extent, crs=proj)
#         im = ax.imshow(data[idx], cmap="Blues", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
#         ax.coastlines(resolution='10m', color='black', linewidth=0.3)
#         ax.add_feature(cfeature.BORDERS, linewidth=0.5)

#         # For the first plot (input), use the original time
#         if idx == 0:
#             time_label = dt.strftime("%Y-%m-%d %H:00")
#         else:
#             # For the other plots, add 12 hours
#             future_dt = dt + timedelta(hours=12)
#             time_label = future_dt.strftime("%Y-%m-%d %H:00")

#         ax.set_title(f"{titles[idx]}\n{time_label}", fontsize=12)
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])), bbox_inches='tight', dpi=300)
#     plt.close(fig)


def visualize_surface_gulf_cartopy(output_deter, output_stoch, target, input, var, step, path, cfg):
    # Crop to region
    output_deter = output_deter[:, 25:-72, 122:-70]
    output_stoch = output_stoch[:, 25:-72, 122:-70]
    target = target[:, 25:-72, 122:-70]
    input = input[:, 25:-72, 122:-70]

    variables = cfg.ERA5_SURFACE_VARIABLES
    var_idx = variables.index(var)

    # Ensure numpy arrays
    output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
    output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # Convert PM values from kg/m³ to µg/m³ (multiply by 1e9)
    if var in ['pm1', 'pm2p5', 'pm10']:
        input = input * 1e9
        output_deter = output_deter * 1e9
        output_stoch = output_stoch * 1e9
        target = target * 1e9

    # Manual color scale
    vmin = 1.0     # µg/m³
    vmax = 50.0    # µg/m³

    proj = ccrs.PlateCarree()
    extent = [30, 60, 10, 40]

    # Plot titles and data
    titles = ['Input', 'Output (Deterministic)', 'Output (Stochastic)', 'Target']
    data = [input[var_idx-4], output_deter[var_idx-4], output_stoch[var_idx-4], target[var_idx-4]]

    dt = datetime.strptime(step, "%Y%m%d%H")
    fig, axs = plt.subplots(1, 4, figsize=(20, 6), subplot_kw={'projection': proj})

    for idx, ax in enumerate(axs):
        ax.set_extent(extent, crs=proj)
        im = ax.imshow(data[idx], cmap="Blues", vmin=vmin, vmax=vmax,
                    extent=extent, transform=proj, origin='upper')
        ax.coastlines(resolution='10m', color='black', linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        time_label = dt.strftime("%Y-%m-%d %H:00") if idx == 0 else (dt + timedelta(hours=12)).strftime("%Y-%m-%d %H:00")
        ax.set_title(f"{titles[idx]}\n{time_label}", fontsize=12)

    # Add shared horizontal colorbar below all subplots
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    if var in ['pm1', 'pm2p5', 'pm10']:
        cbar.set_label("µg/m³", fontsize=12)

    # Skip tight_layout to avoid Cartopy warning
    # plt.tight_layout()

    # Save
    plt.savefig(os.path.join(path, f"{step}_{variables[var_idx]}"), bbox_inches='tight', dpi=300)
    plt.close(fig)




def visualize_surface_MENA_cartopy(output_deter, output_stoch, target, input, var, step, path, cfg):

    # Crop to region
    # output_deter = output_deter[:, 25:-72, 122:-70]
    # output_stoch = output_stoch[:, 25:-72, 122:-70]
    # target = target[:, 25:-72, 122:-70]
    # input = input[:, 25:-72, 122:-70]

    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)

    # Prepare figure
    fig = plt.figure(figsize=(20, 6))

    # Ensure numpy arrays
    output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
    output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # Compute color scale
    combined = np.concatenate([input[var-4].flatten(), target[var-4].flatten(), output_deter[var-4].flatten(), output_stoch[var-4].flatten()])
    vmin = np.percentile(combined, 0)
    vmax = np.percentile(combined, 99)
    data_range = vmax - vmin
    vmin = vmin + 0.15 * data_range
    vmax = vmax - 0.05 * data_range

    print(f"vmin: {vmin}, vmax: {vmax}")

    vmin= 1.1629731773182663e-09
    vmax= 100.3654967896823525e-09

    proj = ccrs.PlateCarree()
    extent = [30, 60, 10, 40]

    # Plot titles and data
    titles = ['Input', 'Output (Deterministic)', 'Output (Stochastic)', 'Target']
    data = [input[var-4], output_deter[var-4], output_stoch[var-4], target[var-4]]

    # Convert step string to datetime
    dt = datetime.strptime(step, "%Y%m%d%H")

    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx + 1, projection=proj)
        ax.set_extent(extent, crs=proj)
        im = ax.imshow(data[idx], cmap="Blues", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
        ax.coastlines(resolution='10m', color='black', linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        # For the first plot (input), use the original time
        if idx == 0:
            time_label = dt.strftime("%Y-%m-%d %H:00")
        else:
            # For the other plots, add 12 hours
            future_dt = dt + timedelta(hours=12)
            time_label = future_dt.strftime("%Y-%m-%d %H:00")

        ax.set_title(f"{titles[idx]}\n{time_label}", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])), bbox_inches='tight', dpi=300)
    plt.close(fig)