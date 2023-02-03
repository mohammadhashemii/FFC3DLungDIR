import torch
import torch.nn.functional as F
from torch.utils import data
import SimpleITK as sitk
import numpy as np
from pathlib import Path

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


def generate_IDs_list_v2(root, case_list: list, n_phases=10, phases= None):
    '''

    this function generates a list containing dictionaries of paired images indices
    Note: it supports patches!!

    accept_reverse_orderL: if True, each patch will be considered for both moving and fixed images 
    '''
    
    if phases is not None:
        phases_list = phases
    else:
        phases_list = range(n_phases)

    IDs_list = []
    for c in case_list:
        for p in phases_list:
            for pp in phases_list:
                patches_dir = Path(root + 'case' + str(c) + '/case' + str(c) + '_patches_w64o8/' )
                patch_filenames = [file.name for file in patches_dir.iterdir() if file.name.startswith('case' + str(c) + '_T' + str(pp) + '0')] 
                for patch_idx in range(len(patch_filenames)):
                    if p != pp:
                        IDs_list += [{
                            'case': c,
                            'fixed_image_phase_num': p,
                            'moving_image_phase_num': pp,
                            'patch_idx' : patch_idx
                        }]

    return IDs_list



def extract_patches_v2(entire_image, win_size=64, overlap_size=8):
    stride = win_size - overlap_size  # stride
    patches = entire_image.unfold(1, win_size, stride).unfold(2, win_size, stride).unfold(3, win_size, stride)
    patches = patches.contiguous().view(patches.size(0), -1, win_size, win_size, win_size)
    
    return patches

    
def extract_patches(entire_image, win_size=64, overlap_size=8, padding=False):
    # entire image shape: (z, x, y)
    z, x, y =  entire_image.shape[0], entire_image.shape[1], entire_image.shape[2]
    patches = []

    if padding:
        n_z =  z // win_size + 1    
        n_x =  x // win_size + 1
        n_y =  y // win_size + 1

        pad_z = (((n_z*win_size) - (n_z-1)*overlap_size) - z) // 2 + 1
        pad_x = (((n_x*win_size) - (n_x-1)*overlap_size) - x) // 2 + 1
        pad_y = (((n_y*win_size) - (n_y-1)*overlap_size) - y) // 2 + 1

        p3d = (pad_y, pad_y, pad_x, pad_y, pad_z, pad_z)
        entire_image = F.pad(entire_image, p3d, "constant", 0)
    else:
        n_z =  z // win_size    
        n_x =  x // win_size
        n_y =  y // win_size

        discard_z = np.abs(z - n_z*win_size - (n_z-1)*overlap_size) // 2
        discard_x = np.abs(x - n_x*win_size - (n_x-1)*overlap_size) // 2
        discard_y = np.abs(y - n_y*win_size - (n_y-1)*overlap_size) // 2
        
        entire_image = entire_image[discard_z:discard_z+(n_z*win_size - (n_z-1)*overlap_size),
                                    discard_x:discard_x+(n_x*win_size - (n_x-1)*overlap_size),
                                    discard_y:discard_y+(n_y*win_size - (n_y-1)*overlap_size)]


    assert x == y # we assume that the height and width of the images are equall
    
    for zz in range(n_z):
        for xx in range(n_x):
            for yy in range(n_x):
                patch = entire_image[zz*win_size-zz*overlap_size:zz*win_size-zz*overlap_size+win_size,
                                            xx*win_size-xx*overlap_size:xx*win_size-xx*overlap_size+win_size,
                                            yy*win_size-yy*overlap_size:yy*win_size-yy*overlap_size+win_size]
                patches.append(patch)

    patches = torch.stack(patches)

    return patches

class DIRLABDataset(data.Dataset):
    def __init__(self, root, case_list: list, patch_size=64, overlap_size=8, phases=[0, 5], transform=None):
        self.root = root
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.transform = transform



        # create a list of image pair IDs
        self.IDs_list = generate_IDs_list_v2(root=root, case_list=case_list, phases=phases)

    def __len__(self):
        return len(self.IDs_list)

    def __getitem__(self, index):
        # select a pair of fixed and moving images (patches)
        ID = self.IDs_list[index]

        # load fixed image data
        im_sitk = sitk.ReadImage(self.root + 'case' + str(ID['case']) + '/case' + str(ID['case']) + '_patches_' + 'w' + str(self.patch_size) + 'o' + str(self.overlap_size)
                                 + '/case' + str(ID['case']) + '_'
                                 + 'T' + str(ID['fixed_image_phase_num']) + '0' 
                                 +  '_patch' + str(ID['patch_idx']) + '.mha')
        fixed_image = torch.Tensor(sitk.GetArrayFromImage(im_sitk))
        # print(torch.max(fixed_image), torch.min(fixed_image))
        # mean, std = torch.mean(fixed_image), torch.std(fixed_image)
        # fixed_image  = (fixed_image - mean) / std

        fixed_image = (fixed_image - torch.min(fixed_image)) / (torch.max(fixed_image) - torch.min(fixed_image))

        
        # load moving image data
        im_sitk = sitk.ReadImage(self.root + 'case' + str(ID['case']) + '/case' + str(ID['case']) + '_patches_' + 'w' + str(self.patch_size) + 'o' + str(self.overlap_size)
                                 + '/case' + str(ID['case']) + '_'
                                 + 'T' + str(ID['moving_image_phase_num']) + '0' 
                                 +  '_patch' + str(ID['patch_idx']) + '.mha')
        moving_image = torch.Tensor(sitk.GetArrayFromImage(im_sitk))
        # mean, std = torch.mean(moving_image), torch.std(moving_image)
        # moving_image   = (moving_image - mean) / std

        moving_image = (moving_image - torch.min(moving_image)) / (torch.max(moving_image) - torch.min(moving_image))



        paired_patches = torch.stack([fixed_image, moving_image], dim=0) # (c=2, d, h, w)

        return paired_patches, ID

        
class CREATISDataset(data.Dataset):
    def __init__(self, root, case_list: list, patch_size=64, overlap_size=8, transform=None):
        self.root = root
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.transform = transform

        # create a list of image pair IDs
        self.IDs_list = generate_IDs_list_v2(root=root, case_list=case_list, n_phases=10)

    def __len__(self):
        return len(self.IDs_list)

    def __getitem__(self, index):
        # select a pair of fixed and moving images (patches)
        ID = self.IDs_list[index]

        # load fixed image data
        im_sitk = sitk.ReadImage(self.root + 'case' + str(ID['case']) + '/case' + str(ID['case']) + '_patches_' + 'w' + str(self.patch_size) + 'o' + str(self.overlap_size)
                                 + '/case' + str(ID['case']) + '_'
                                 + 'T' + str(ID['fixed_image_phase_num']) + '0' 
                                 +  '_patch' + str(ID['patch_idx']) + '.mha')
        fixed_image = torch.Tensor(sitk.GetArrayFromImage(im_sitk))
        # mean, std = torch.mean(fixed_image), torch.std(fixed_image)
        
        # fixed_image  = (fixed_image - mean) / std

        fixed_image = (fixed_image - torch.min(fixed_image)) / (torch.max(fixed_image) - torch.min(fixed_image))

        
        # load moving image data
        im_sitk = sitk.ReadImage(self.root + 'case' + str(ID['case']) + '/case' + str(ID['case']) + '_patches_' + 'w' + str(self.patch_size) + 'o' + str(self.overlap_size)
                                 + '/case' + str(ID['case']) + '_'
                                 + 'T' + str(ID['moving_image_phase_num']) + '0' 
                                 +  '_patch' + str(ID['patch_idx']) + '.mha')
        moving_image = torch.Tensor(sitk.GetArrayFromImage(im_sitk))
        # mean, std = torch.mean(moving_image), torch.std(moving_image)
        # moving_image   = (moving_image - mean) / std

        moving_image = (moving_image - torch.min(moving_image)) / (torch.max(moving_image) - torch.min(moving_image))



        paired_patches = torch.stack([fixed_image, moving_image], dim=0) # (c=2, d, h, w)

        return paired_patches
