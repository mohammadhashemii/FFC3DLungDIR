from fuse_patches import fuse_dvfs
import os
import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors


def plot_intensity_differences(case_num, exp):
    fig, ax = plt.subplots(len(case_num), 5, figsize=(30, 30))
    plt.tight_layout()
    plt.rc('font', size=18)

    # custom_cmap = LinearSegmentedColormap.from_list('', ['red', 'white', 'darkblue'])
    custom_cmap = LinearSegmentedColormap.from_list('', ['black', '#e7337e'])
    
    
    for c in range(len(case_num)):
        # fixed img
        im_sitk = sitk.ReadImage('./data/DIR-Lab/4DCT/mha_cropped/case'+str(case_num[c])+'/case'+str(case_num[c])+'_T00.mha')
        fixed_img = sitk.GetArrayFromImage(im_sitk).astype(np.float64)
        d, h, w = fixed_img.shape
        fixed_img = fixed_img[d // 2, :]
        fixed_img *= 255/fixed_img.max() 
        ax[c,0].imshow(fixed_img, cmap='gray')
        ax[c,0].set_axis_off()
        # if c == 1:
        #     ax[c-1,0].set_title("Fixed Image", fontweight="bold")

        
        # moving img
        im_sitk = sitk.ReadImage('./data/DIR-Lab/4DCT/mha_cropped/case'+str(case_num[c])+'/case'+str(case_num[c])+'_T50.mha')
        moving_img = sitk.GetArrayFromImage(im_sitk).astype(np.float64)
        d, h, w = moving_img.shape
        moving_img = moving_img[d // 2, :]
        moving_img *= 255/moving_img.max() 
        ax[c,1].imshow(moving_img, cmap='gray')
        ax[c,1].set_axis_off()
        # if c == 1:
        #     ax[c-1,1].set_title("Moving Image", fontweight="bold")

        print(moving_img.shape)

        

        # difference before registration 
        # diff = fixed_img - moving_img
        diff = np.abs(fixed_img - moving_img)
        # divnorm=colors.TwoSlopeNorm(vmin=-256., vcenter=0., vmax=256.)
        divnorm=colors.TwoSlopeNorm(vmin=0,vcenter=128, vmax=256.)
        diff_colored = ax[c, 2].imshow(diff, cmap=custom_cmap, norm=divnorm)
        ax[c,2].set_axis_off()
        # if c == 1:
        #     ax[c-1,2].set_title("Before Registration", fontweight="bold")


        # registered image
        registered_volume = torch.load("./experiments/" + "exp" + str(exp) + "_figs/" + "epoch35_case" + str(case_num[c]) + "_registered_volume.pt")
        print(registered_volume.shape)
        registered_img = registered_volume[d // 2, :]
        registered_img *= 255/registered_img.max() 
        ax[c,3].imshow(registered_img, cmap='gray')
        ax[c,3].set_axis_off()
        # if c == 1:
        #     ax[c-1,3].set_title("Warped Image", fontweight="bold")

        # difference after registration 
        # diff = torch.from_numpy(fixed_img) - registered_img
        diff = np.abs(torch.from_numpy(fixed_img) - registered_img)
        # divnorm=colors.TwoSlopeNorm(vmin=-256., vcenter=0., vmax=256.)
        divnorm=colors.TwoSlopeNorm(vmin=0, vcenter=128, vmax=256.)
        diff_colored = ax[c, 4].imshow(diff, cmap=custom_cmap, norm=divnorm)
        ax[c,4].set_axis_off()
        # if c == 1:
        #     ax[c-1,4].set_title("After Registration", fontweight="bold")
        

    # plt.colorbar(diff_colored, ax=ax, orientation='horizontal', ticks=[-256,-128,-64,0, 64, 128, 256])
    plt.colorbar(diff_colored, ax=ax, orientation='horizontal', ticks=[0,32, 64, 128, 256])

    plt.show()

plot_intensity_differences(case_num=[1, 2], exp=15)