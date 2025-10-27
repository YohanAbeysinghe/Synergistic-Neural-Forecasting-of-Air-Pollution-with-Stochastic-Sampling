import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_surface_mena(output_deter, output_stoch, target, input, var, step, path, cfg):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(24, 2))

    # Ensure all inputs are NumPy arrays
    # Compute vmin and vmax across all relevant data for consistent color scaling
    output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
    output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # Use percentiles for robust color scaling, which ignores extreme outliers
    combined = np.concatenate([input[var-4].flatten(), target[var-4].flatten(), output_deter[var-4].flatten(), output_stoch[var-4].flatten()])
    vmin = np.percentile(combined, 0)
    vmax = np.percentile(combined, 97)


    ax1 = fig.add_subplot(161)
    plot1 = ax1.imshow(input[var-4, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot1 = ax1.imshow(input[var, :, :], cmap="RdBu")
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('input')

    # New subplot for the sliced region of 'input'
    ax2 = fig.add_subplot(162)
    plot2 = ax2.imshow(output_deter[var-4, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot2 = ax2.imshow(input[var, :, :], cmap="RdBu")
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('output_deter')

    ax3 = fig.add_subplot(163)
    plot3 = ax3.imshow(output_stoch[var-4, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot3 = ax3.imshow(target[var, :, :], cmap="RdBu")
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('output_stoch')

    ax4 = fig.add_subplot(164)
    plot4 = ax4.imshow(target[var-4, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    # plot4 = ax4.imshow(output[var, :, :], cmap="RdBu")
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('target')

    bias_deter = output_deter[var-4, :, :] - target[var-4, :, :]
    bias_stoch = output_stoch[var-4, :, :] - target[var-4, :, :]

    all_bias = np.concatenate([bias_deter.flatten(), bias_stoch.flatten()])
    bias_limit = np.percentile(np.abs(all_bias), 99)  # Get 95th percentile of absolute bias

    bias_min = -bias_limit
    bias_max = bias_limit

    # bias_max = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 0)
    # bias_min = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 95)

    ax5 = fig.add_subplot(165)
    plot5 = ax5.imshow(bias_deter, cmap="RdBu_r", vmin=bias_min, vmax=bias_max)
    plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
    ax5.title.set_text('bias_output_deter')

    ax5 = fig.add_subplot(166)
    plot5 = ax5.imshow(bias_stoch, cmap="RdBu_r", vmin=bias_min, vmax=bias_max)
    plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
    ax5.title.set_text('bias_output_stoch')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
    plt.close(fig)


def visualize_surface_mena_cartopy(output_deter, output_stoch, target, input, var, step, path, cfg):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(24, 3))

    # Ensure all inputs are NumPy arrays
    # Compute vmin and vmax across all relevant data for consistent color scaling
    output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
    output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # Use percentiles for robust color scaling, which ignores extreme outliers
    combined = np.concatenate([input[var-4].flatten(), target[var-4].flatten(), output_deter[var-4].flatten(), output_stoch[var-4].flatten()])
    vmin = np.percentile(combined, 0)
    vmax = np.percentile(combined, 99)


    proj = ccrs.PlateCarree()
    lat_start_idx = 175
    lat_end_idx = 392
    lon_start_idx = 718
    lon_end_idx = 1030
    lat_res = 0.25
    lon_res = 0.25
    lat_max = 90 - lat_start_idx * lat_res
    lat_min = 90 - lat_end_idx * lat_res
    lon_min = -180 + lon_start_idx * lon_res
    lon_max = -180 + lon_end_idx * lon_res
    extent = [lon_min, lon_max, lat_min, lat_max]


    ax1 = fig.add_subplot(161, projection=proj)
    ax1.set_extent(extent, crs=proj)
    plot1 = ax1.imshow(input[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax1.coastlines(resolution='10m', color='black', linewidth=1)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('input')

    ax2 = fig.add_subplot(162, projection=proj)
    ax2.set_extent(extent, crs=proj)
    plot2 = ax2.imshow(output_deter[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax2.coastlines(resolution='10m', color='black', linewidth=1)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('output_deter')

    ax3 = fig.add_subplot(163, projection=proj)
    ax3.set_extent(extent, crs=proj)
    plot3 = ax3.imshow(output_stoch[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax3.coastlines(resolution='10m', color='black', linewidth=1)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('output_stoch')

    ax4 = fig.add_subplot(164, projection=proj)
    ax4.set_extent(extent, crs=proj)
    plot4 = ax4.imshow(target[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax4.coastlines(resolution='10m', color='black', linewidth=1)
    ax4.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('target')

    bias_deter = output_deter[var-4, :, :] - target[var-4, :, :]
    bias_stoch = output_stoch[var-4, :, :] - target[var-4, :, :]

    all_bias = np.concatenate([bias_deter.flatten(), bias_stoch.flatten()])
    bias_limit = np.percentile(np.abs(all_bias), 99)  # Get 95th percentile of absolute bias

    bias_min = -bias_limit
    bias_max = bias_limit

    # bias_max = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 0)
    # bias_min = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 95)

    ax5 = fig.add_subplot(165, projection=proj)
    ax5.set_extent(extent, crs=proj)
    plot5 = ax5.imshow(bias_deter, cmap="RdBu_r", vmin=bias_min, vmax=bias_max, extent=extent, transform=proj, origin='upper')
    ax5.coastlines(resolution='10m', color='black', linewidth=1)
    ax5.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
    ax5.title.set_text('bias_output_deter')

    ax6 = fig.add_subplot(166, projection=proj)
    ax6.set_extent(extent, crs=proj)
    plot6 = ax6.imshow(bias_stoch, cmap="RdBu_r", vmin=bias_min, vmax=bias_max, extent=extent, transform=proj, origin='upper')
    ax6.coastlines(resolution='10m', color='black', linewidth=1)
    ax6.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot6, ax=ax6, fraction=0.05, pad=0.05)
    ax6.title.set_text('bias_output_stoch')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
    plt.close(fig)




def visualize_surface_gulf_cartopy(output_deter, output_stoch, target, input, var, step, path, cfg):

    output_deter = output_deter[:, 25:-72, 122:-70]
    output_stoch = output_stoch[:, 25:-72, 122:-70]
    target = target[:, 25:-72, 122:-70]
    input = input[:, 25:-72, 122:-70]

    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(24, 3))

    # Ensure all inputs are NumPy arrays
    # Compute vmin and vmax across all relevant data for consistent color scaling
    output_deter = output_deter.detach().cpu().numpy() if not isinstance(output_deter, np.ndarray) else output_deter
    output_stoch = output_stoch.detach().cpu().numpy() if not isinstance(output_stoch, np.ndarray) else output_stoch
    target = target.detach().cpu().numpy() if not isinstance(target, np.ndarray) else target
    input = input.detach().cpu().numpy() if not isinstance(input, np.ndarray) else input

    # Use percentiles for robust color scaling, which ignores extreme outliers
    combined = np.concatenate([input[var-4].flatten(), target[var-4].flatten(), output_deter[var-4].flatten(), output_stoch[var-4].flatten()])
    vmin = np.percentile(combined, 0)
    vmax = np.percentile(combined, 99)
    data_range = vmax - vmin

    # Extend the range by 5% on both sides
    vmin = vmin + 0.15 * data_range
    vmax = vmax - 0.05 * data_range


    proj = ccrs.PlateCarree()
    extent = [30, 60, 10, 40]


    ax1 = fig.add_subplot(161, projection=proj)
    ax1.set_extent(extent, crs=proj)
    plot1 = ax1.imshow(input[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax1.coastlines(resolution='10m', color='black', linewidth=1)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot1, ax=ax1, fraction=0.05, pad=0.05)
    ax1.title.set_text('input')

    ax2 = fig.add_subplot(162, projection=proj)
    ax2.set_extent(extent, crs=proj)
    plot2 = ax2.imshow(output_deter[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax2.coastlines(resolution='10m', color='black', linewidth=1)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot2, ax=ax2, fraction=0.05, pad=0.05)
    ax2.title.set_text('output_deter')

    ax3 = fig.add_subplot(163, projection=proj)
    ax3.set_extent(extent, crs=proj)
    plot3 = ax3.imshow(output_stoch[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax3.coastlines(resolution='10m', color='black', linewidth=1)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot3, ax=ax3, fraction=0.05, pad=0.05)
    ax3.title.set_text('output_stoch')

    ax4 = fig.add_subplot(164, projection=proj)
    ax4.set_extent(extent, crs=proj)
    plot4 = ax4.imshow(target[var-4, :, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, transform=proj, origin='upper')
    ax4.coastlines(resolution='10m', color='black', linewidth=1)
    ax4.add_feature(cfeature.BORDERS, linewidth=0.5)
    plt.colorbar(plot4, ax=ax4, fraction=0.05, pad=0.05)
    ax4.title.set_text('target')

    bias_deter = output_deter[var-4, :, :] - target[var-4, :, :]
    bias_stoch = output_stoch[var-4, :, :] - target[var-4, :, :]

    all_bias = np.concatenate([bias_deter.flatten(), bias_stoch.flatten()])
    bias_limit = np.percentile(np.abs(all_bias), 100) + np.percentile(np.abs(all_bias), 10)  # Get 95th percentile of absolute bias

    bias_min = -bias_limit
    bias_max = bias_limit

    # bias_max = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 0)
    # bias_min = np.percentile(np.concatenate([bias_deter.flatten(), bias_stoch.flatten()]), 95)

    ax5 = fig.add_subplot(165, projection=proj)
    ax5.set_extent(extent, crs=proj)
    plot5 = ax5.imshow(bias_deter, cmap="RdBu", vmin=bias_min, vmax=bias_max, extent=extent, transform=proj, origin='upper')
    ax5.coastlines(resolution='10m', color='black', linewidth=1)
    ax5.add_feature(cfeature.BORDERS, linewidth=0.5)
    # plt.colorbar(plot5, ax=ax5, fraction=0.05, pad=0.05)
    ax5.title.set_text('bias_output_deter')

    ax6 = fig.add_subplot(166, projection=proj)
    ax6.set_extent(extent, crs=proj)
    plot6 = ax6.imshow(bias_stoch, cmap="RdBu", vmin=bias_min, vmax=bias_max, extent=extent, transform=proj, origin='upper')
    ax6.coastlines(resolution='10m', color='black', linewidth=1)
    ax6.add_feature(cfeature.BORDERS, linewidth=0.5)
    # plt.colorbar(plot6, ax=ax6, fraction=0.05, pad=0.05)
    ax6.title.set_text('bias_output_stoch')

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, '{}_{}'.format(step, variables[var])))
    plt.close(fig)