import torch
import numpy as np
import SimpleITK as sitk
from math import ceil
# import view3d_image

original_dirlab_volume_sizes = [[256, 256, 94],
                                [256, 256, 112],
                                [256, 256, 104],
                                [256, 256, 99],
                                [256, 256, 106],
                                [350, 350, 128],
                                [350, 350, 136],
                                [300, 300, 128],
                                [300, 300, 128],
                                [300, 300, 120]]

def fuse_patches(patches, case_num, win_size=64, overlap_size=8):
    original_size = original_dirlab_volume_sizes[case_num - 1]
    if case_num in [6, 7]:
        n_x, n_y, n_z = 6, 6, 3
        overlap_size_x_y = ceil((win_size * n_x - original_size[0]) / (n_x - 1))
        fused_img = torch.zeros((original_size[2], win_size * n_x - (n_x-1)*overlap_size_x_y, win_size * n_x - (n_x-1)*overlap_size_x_y), dtype=torch.float16)
    elif case_num > 5 and case_num != 7 and case_num != 6:
        n_x, n_y, n_z = 5, 5, 3
        overlap_size_x_y = (win_size * n_x - original_size[0]) // (n_x - 1)
        fused_img = torch.zeros((original_size[2], original_size[0], original_size[1]), dtype=torch.float16)
    else:
        n_x, n_y, n_z = 5, 5, 2
        overlap_size_x_y = (win_size * n_x - original_size[0]) // (n_x - 1)
        fused_img = torch.zeros((original_size[2], original_size[0], original_size[1]), dtype=torch.float16)
    

    overlap_size_z = (win_size * n_z - original_size[2]) // (n_z - 1)

    patch_idx = 0
    for z in range(n_z):
        for x in range(n_x):
            for y in range(n_y):
                fused_img[z*win_size-z*overlap_size_z:(z+1)*win_size-z*overlap_size_z, 
                          x*win_size-x*overlap_size_x_y:(x+1)*win_size-x*overlap_size_x_y, 
                          y*win_size-y*overlap_size_x_y:(y+1)*win_size-y*overlap_size_x_y] = patches[patch_idx, :]
                patch_idx += 1

    return fused_img


def fuse_dvfs(patches, case_num, win_size=64, overlap_size=8):
    original_size = original_dirlab_volume_sizes[case_num - 1]
    if case_num in [6, 7]:
        n_x, n_y, n_z = 6, 6, 3
        overlap_size_x_y = ceil((win_size * n_x - original_size[0]) / (n_x - 1))
        fused_dvf = torch.zeros((3, original_size[2], win_size * n_x - (n_x-1)*overlap_size_x_y, win_size * n_x - (n_x-1)*overlap_size_x_y), dtype=torch.float16)
    elif case_num > 5 and case_num != 7 and case_num != 6:
        n_x, n_y, n_z = 5, 5, 3
        overlap_size_x_y = (win_size * n_x - original_size[0]) // (n_x - 1)
        fused_dvf = torch.zeros((3, original_size[2], original_size[0], original_size[1]), dtype=torch.float16)
    else:
        n_x, n_y, n_z = 5, 5, 2
        overlap_size_x_y = (win_size * n_x - original_size[0]) // (n_x - 1)
        fused_dvf = torch.zeros((3, original_size[2], original_size[0], original_size[1]), dtype=torch.float16)
    

    overlap_size_z = (win_size * n_z - original_size[2]) // (n_z - 1)

    patch_idx = 0
    for z in range(n_z):
        for x in range(n_x):
            for y in range(n_y):
                fused_dvf[:, z*win_size-z*overlap_size_z:(z+1)*win_size-z*overlap_size_z, 
                          x*win_size-x*overlap_size_x_y:(x+1)*win_size-x*overlap_size_x_y, 
                          y*win_size-y*overlap_size_x_y:(y+1)*win_size-y*overlap_size_x_y] = patches[patch_idx, :]
                patch_idx += 1

    return fused_dvf
