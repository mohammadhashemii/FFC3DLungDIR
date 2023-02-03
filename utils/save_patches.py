import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data
import SimpleITK as sitk
import image_processing as ip
import os
from math import ceil
import view3d_image

def generate_IDs_list(case_list: list, n_phases=10, phases= None):
    '''

    this function generates a list containing dictionaries of paired images indices
    '''
    
    if phases is not None:
        phases_list = phases
    else:
        phases_list = range(n_phases)

    IDs_list = []
    for c in case_list:
        for p in phases_list:
            for pp in phases_list:
                if p != pp:
                    IDs_list += [{
                        'case': c,
                        'fixed_image_phase_num': p,
                        'moving_image_phase_num': pp
                    }]

    return IDs_list

def extract_patches_v2(dataset_name, case_num, entire_image, win_size=64, overlap_size=8):
    if dataset_name == "DIR-Lab":
        if case_num < 6:
            n_z = 2 # number of patches over z axis
            overlap_size_z = (win_size * n_z - entire_image.shape[1]) // (n_z - 1)
        else:
            n_z = 3 # number of patches over z axis
            overlap_size_z = (win_size * n_z - entire_image.shape[1]) // (n_z - 1)

        if case_num in [6, 7]:
            n_x = n_y = 6 # number of patches over x, y axis
            overlap_size_x_y = ceil((win_size * n_x - entire_image.shape[2]) / (n_x - 1))
        else:
            n_x = n_y = 5 # number of patches over x, y axis
            overlap_size_x_y = (win_size * n_x - entire_image.shape[2]) // (n_x - 1)
    else:
        overlap_size_z = overlap_size
        overlap_size_x_y = overlap_size

    stride_z = win_size - overlap_size_z # stride z
    stride_x_y = win_size - overlap_size_x_y  # stride x, y
    patches = entire_image.unfold(1, win_size, stride_z).unfold(2, win_size, stride_x_y).unfold(3, win_size, stride_x_y)
    patches = patches.contiguous().view(patches.size(0), -1, win_size, win_size, win_size)

    return patches

def save_patches_as_tensors(dataset_name, root, case_list: list, phases: list, ext='.mha', patch_size=64, overlap_size=8):
    for c in case_list:
        patches_dir = root + 'case' + str(c) + '/case' + str(c) + '_patches_'+ 'w' + str(patch_size) + 'o' + str(overlap_size)+ '/'
        if not os.path.exists(patches_dir):
                os.mkdir(patches_dir)

        for p in phases:
            im_sitk = sitk.ReadImage(root + 'case' + str(c)
                                 + '/case' + str(c) + '_'
                                 + 'T' + str(p) + '0' + '.mha')
            entire_img = torch.Tensor(sitk.GetArrayFromImage(im_sitk)).unsqueeze(0)
            patches = extract_patches_v2(dataset_name, c, entire_img,
                                         win_size=patch_size, overlap_size=overlap_size) # (1, patches, d, h, w)

            for patch_idx in range(patches.shape[1]):
                patch_sitk = ip.array_to_sitk(patches[0, patch_idx])
                sitk.WriteImage(patch_sitk, patches_dir + 'case' + str(c) + '_T' + str(p) + '0' + '_patch' + str(patch_idx) + ext)
                
            # view3d_image.view3d_image(patches[0, 10, :, :, :], slice_axis=0) # moving patch 
        print("{} Patches of case {} is saved!".format(patches.shape[1], str(c)))    

if __name__ == '__main__':
    save_patches_as_tensors("DIR-Lab", root="./data/DIR-Lab/4DCT/mha_cropped/",
                             case_list=[6, 7],
                             phases=range(10), ext='.mha', patch_size=64, overlap_size=8)
