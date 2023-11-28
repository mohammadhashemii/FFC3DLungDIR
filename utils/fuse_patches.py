import torch
import numpy as np
import SimpleITK as sitk
from math import ceil
from utils.view3d_image import *


def fuse_dvf(patches, patch_size=(64, 64, 64), overlap=(8, 8, 8), output_size = None):
    patch_d, patch_h, patch_w = patch_size
    overlap_d, overlap_h, overlap_w = overlap

    # Calculate the dimensions of the final fused array
    D, H, W = output_size


    # Initialize the fused array with zeros
    fused_array = torch.zeros((3, D, H, W))
    count_array = torch.zeros((3, D, H, W))

    # Iterate through the patches and fuse them into the final array
    patch_count = 0
    for d in range(0, D, patch_d - overlap_d):
        for h in range(0, H, patch_h - overlap_h):
            for w in range(0, W, patch_w - overlap_w):
                patch = patches[patch_count]

                # Calculate the end indices for the current patch
                d_end = min(d + patch_d, D)
                h_end = min(h + patch_h, H)
                w_end = min(w + patch_w, W)

                d_start = d
                h_start = h
                w_start = w

                if d_end - d < patch_d:
                    d_start = d_end - patch_d
                    d_end +=1

                if h_end - h < patch_h:
                    h_start = h_end - patch_h
                    h_end +=1

                if w_end - w < patch_w:
                    w_start = w_end - patch_w
                    w_end +=1


                # Fuse the current patch into the final array
                
                fused_array[:, d_start:d_end, h_start:h_end, w_start:w_end] += patch
                count_array[:, d_start:d_end, h_start:h_end, w_start:w_end] += 1

                patch_count += 1

    
    average_array = torch.divide(fused_array, count_array)

    return average_array


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
