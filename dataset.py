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


def extract_patch(fixed_image_array, moving_image_array, patch_size, selection_type='random'):
    D, H, W = fixed_image_array.shape
    if selection_type == 'random':
        # Randomly choose the starting coordinates for the patch from uniform distribution
        d_start = np.random.randint(0, D - patch_size + 1)
        h_start = np.random.randint(0, H - patch_size + 1)
        w_start = np.random.randint(0, W - patch_size + 1)

    elif selection_type == 'normal':
        # Define standard deviation to ensure 99.9% of samples occur within [0,(*)-patch_size] for (*) = D, H, W
        d_sigma = (D - patch_size)/6
        h_sigma = (H - patch_size)/6
        w_sigma = (W - patch_size)/6
        d_mu = (D - patch_size)/2
        h_mu = (H - patch_size)/2
        w_mu = (W - patch_size)/2

        # Randomly choose the starting coordinates for the patch from normal distribution
        d_start = d_sigma * np.random.randn() + d_mu
        h_start = h_sigma * np.random.randn() + h_mu
        w_start = w_sigma * np.random.randn() + w_mu

        # Round index to nearest integer
        d_start = int(round(d_start))
        h_start = int(round(h_start))
        w_start = int(round(w_start))

        # Ensure valid index is chosen
        if d_start < 0:
            d_start = 0
        if d_start >= D - patch_size + 1:
            d_start = D - patch_size

        if h_start < 0:
            h_start = 0
        if h_start >= H - patch_size + 1:
            h_start = H - patch_size

        if w_start < 0:
            w_start = 0
        if w_start >= W - patch_size + 1:
            w_start = W - patch_size

    else:
        raise ValueError("Unsupported selection_type")

    # Extract patches from both images
    fixed_patch = fixed_image_array[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]
    moving_patch = moving_image_array[d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size]

    # Stack the patches along the first dimension
    stacked_patch = np.stack([fixed_patch, moving_patch], axis=0)

    return stacked_patch



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
    def __init__(self, root, case_list: list, patch_size=64, patch_selection='normal', transform=None):
        self.root = root
        self.patch_size = patch_size
        self.patch_selection = patch_selection
        self.transform = transform

        # create a list of image pair IDs
        self.IDs_list = generate_IDs_list(case_list=case_list, n_phases=10)

    def __len__(self):
        return len(self.IDs_list)

    def __getitem__(self, index):
        # select a pair of fixed and moving images (patches)
        ID = self.IDs_list[index]

        # load fixed image data
        im_sitk = sitk.ReadImage(self.root + str(ID['case']) + '/' + str(ID['fixed_image_phase_num']) + '0' '_R.mha')
        fixed_image = torch.Tensor(sitk.GetArrayFromImage(im_sitk))
        fixed_image = (fixed_image - torch.min(fixed_image)) / (torch.max(fixed_image) - torch.min(fixed_image))

        # load moving image data
        im_sitk = sitk.ReadImage(self.root + str(ID['case']) + '/' + str(ID['moving_image_phase_num']) + '0' '_R.mha')
        moving_image = torch.Tensor(sitk.GetArrayFromImage(im_sitk))

        moving_image = (moving_image - torch.min(moving_image)) / (torch.max(moving_image) - torch.min(moving_image))


        # extract a single pair of patch
        paired_patches = extract_patch(fixed_image, moving_image, patch_size=self.patch_size, selection_type=self.patch_selection)

        return paired_patches
