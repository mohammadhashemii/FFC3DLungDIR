from fuse_patches import fuse_dvfs
import os
import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors

def plot_dvf(dvf):
    
    displ = sitk.TransformToDisplacementField(dvf, 
                                  sitk.sitkVectorFloat64)
                                #   image.GetSize(),
                                #   image.GetOrigin(),
                                #   image.GetSpacing(),
                                #   image.GetDirection())
    det = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(displ))
    vectors = sitk.GetArrayFromImage(displ)
    
    x, y = det.shape
    X, Y = np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1), indexing='ij')
    U = vectors[:,:,1]
    V = vectors[:,:,2]    
    plt.imshow(det, cmap='jet')

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_dvf_jacobian(dvf, case_num, exp):
    fig, ax = plt.subplots(1, 4, figsize=(30, 20))
    plt.tight_layout()

    custom_cmap = LinearSegmentedColormap.from_list('', ['orange', 'white', 'darkblue'])
    
    # fixed img
    im_sitk = sitk.ReadImage('./data/DIR-Lab/4DCT/mha_cropped/case'+str(case_num)+'/case'+str(case_num)+'_T00.mha')
    fixed_img = sitk.GetArrayFromImage(im_sitk).astype(np.float64)
    d, h, w = fixed_img.shape
    fixed_img = fixed_img[d // 2, :]
    fixed_img *= 255/fixed_img.max() 
    ax[0].imshow(fixed_img, cmap='gray')
    ax[0].set_axis_off()

    
    # moving img
    im_sitk = sitk.ReadImage('./data/DIR-Lab/4DCT/mha_cropped/case'+str(case_num)+'/case'+str(case_num)+'_T50.mha')
    moving_img = sitk.GetArrayFromImage(im_sitk).astype(np.float64)
    d, h, w = moving_img.shape
    moving_img = moving_img[d // 2, :]
    moving_img *= 255/moving_img.max() 
    ax[2].imshow(moving_img, cmap='gray')
    ax[2].set_axis_off()


    # registered image
    registered_volume = torch.load("./experiments/" + "exp" + str(exp) + "_figs/" + "epoch12_case" + str(case_num) + "_registered_volume.pt")
    registered_img = registered_volume[d // 2, :]
    registered_img *= 255/registered_img.max() 
    ax[1].imshow(registered_img, cmap='gray')
    ax[1].set_axis_off()

    #jacobian

    # dvf shape must be: (d, h, w, 3)
    #random 3D displacement field in numpy array
    dvf = dvf.cpu().detach().numpy().astype(np.float64)
    print(dvf.shape)

    #Convert the numpy array to a 256x256x20 image with each pixel 
    #being a 3D vector and compute the jacobian determinant volume
    sitk_displacement_field = sitk.GetImageFromArray(dvf, isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian_det_np_arr = sitk.GetArrayViewFromImage(jacobian_det_volume)

    # Iterating over the slices
    
    det = jacobian_det_np_arr[dvf.shape[0] // 2,:,:]

    divnorm=colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=2.)
    det_plot = ax[3].imshow(det, cmap=custom_cmap, norm=divnorm)
    ax[3].set_axis_off()
    plt.colorbar(det_plot, ax=ax, orientation='horizontal', ticks=[0,0.5,1, 1.5, 2])

    plt.show()


EXPERIMENT = 18
case_num = 5

# DVF patch fusion
DVF_dir = "./experiments/" + "exp" + str(EXPERIMENT) + "_DVF/"
DVF_patches_filenames = os.listdir(DVF_dir)
fixed_img_phase = '00'
moving_img_phase = '50'
DVF_filenames = []
for f in DVF_patches_filenames:
    if "f" + fixed_img_phase + "_m" + moving_img_phase in f and "case" + str(case_num) + "_" in f:
        DVF_filenames.append(f)


dvfs = []
for i in range(len(DVF_filenames)):
    patch_dvf = torch.load(DVF_dir + "DVF_case" + str(case_num) 
                                    + "_f" + fixed_img_phase  # fixed image phase number
                                    + "_m" + moving_img_phase # moving image phase number
                                    + "_patch" + str(i) + ".pt", map_location=torch.device('cpu'))
    dvfs.append(patch_dvf.squeeze(0))
    

dvfs = torch.stack(dvfs, 0)
fused_dvf = fuse_dvfs(dvfs, case_num=case_num, win_size=64, overlap_size=8)


# plot jacobian
_, d, h, w = fused_dvf.size()
dvf = fused_dvf[0:2, d // 2, :]


# plot_dvf_jacobian(fused_dvf.permute(1, 2, 3, 0), case_num=case_num, exp=18)





print(fused_dvf.shape)

fig, ax = plt.subplots()
ax.set_facecolor('black')
ax.margins(x=0)
ax.margins(y=0)
n_samples = 16
grid_x,grid_y = np.meshgrid(np.linspace(0,255,n_samples),np.linspace(0,255,n_samples))
# plot_grid(grid_x,grid_y, ax=ax,  color="lightgrey")



dvf_x, dvf_y = dvf[1, :], dvf[0, :]

# print(grid_y)


f = lambda x,y : ( torch.from_numpy(x) + dvf_x, torch.from_numpy(y) + dvf_y)
grid_x,grid_y = np.meshgrid(np.linspace(0,255,256),np.linspace(0,255,256))
distx, disty = f(grid_x,grid_y)


grid_x,grid_y = np.meshgrid(np.linspace(0,255,n_samples),np.linspace(0,255,n_samples))

distx = torch.index_select(distx, 1, torch.from_numpy(grid_x[0].astype(int)))
distx = torch.index_select(distx, 0, torch.from_numpy(grid_x[0].astype(int)))

disty = torch.index_select(disty, 0, torch.from_numpy(grid_y[:, 0].astype(int)))
disty = torch.index_select(disty, 1, torch.from_numpy(grid_y[:, 0].astype(int)))

plot_grid(distx, disty, ax=ax, color="yellow")






im_sitk = sitk.ReadImage('./data/DIR-Lab/4DCT/mha_cropped/case5/case5_T50.mha')

im = sitk.GetArrayFromImage(im_sitk)

ax.imshow(im[d // 2, :], cmap='gray')

plt.show()