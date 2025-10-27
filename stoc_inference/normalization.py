import numpy as np

def normBack(surface, statistics):
    surface_mean, surface_std, upper_mean, upper_std = (statistics[0], statistics[1], statistics[2], statistics[3])
    surface_mean = surface_mean[:, 4:7, :, :]
    surface_std = surface_std[:, 4:7, :, :]
    surface = surface * surface_std + surface_mean
    return surface
