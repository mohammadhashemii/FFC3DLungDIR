import torch
from torch.utils.data import DataLoader
from dataset import  DIRLABDataset
from utils.yaml_reader import load_config
import SimpleITK as sitk
from utils.fuse_patches import *
from utils.TRE import compute_TRE, create_registered_landmark
from models.model import FFCResNetGenerator
from models.SpatialTransformerNetwork import SpatialTransformation
import numpy as np
from tqdm import tqdm
import os
from utils.view3d_image import *
torch.manual_seed(42)


resampled_dirlab_volume_sizes = [[234, 248, 248],
                                [279, 296, 296],
                                [259, 294, 294],
                                [247, 289, 289],
                                [264, 281, 281],
                                [319, 496, 496],
                                [339, 496, 496],
                                [319, 496, 496],
                                [319, 496, 496],
                                [299, 496, 496]]

def test_per_case(experiment, epoch_idx, case_num=1, phases_list=[0, 5], save_figs=False):
    
    # Loading configs
    test_configs = load_config('./configs/inference_settings.yaml')
    model_configs = load_config('./configs/FFCResnetGenerator_settings.yaml')

    # Creating datasets
    test_dataset = DIRLABDataset(root=test_configs['test_data_path'],
                                 case_list=[case_num],
                                 phases=phases_list) # maximum inhale and exhale
    test_loader = DataLoader(test_dataset, **test_configs['data_loader'])

    # Model defining
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    FFCGenerator = FFCResNetGenerator(input_nc=model_configs['input_nc'],
                                    output_nc=model_configs['output_nc'],
                                    n_downsampling=model_configs['n_downsampling'],
                                    n_blocks=model_configs['n_blocks'],
                                    init_conv_kwargs=model_configs['init_conv_kwargs'],
                                    downsample_conv_kwargs=model_configs['downsample_conv_kwargs'],
                                    resnet_conv_kwargs=model_configs['resnet_conv_kwargs']).to(device)
    stn = SpatialTransformation(use_gpu=False)

    # Load the weights
    generator_weights = "saved_ours/exp" + experiment + "_epoch" + str(epoch_idx) + "_gen.pth"
    stn_weights = "saved_ours/exp" + experiment + "_epoch" + str(epoch_idx) + "_stn.pth"
    
    FFCGenerator.load_state_dict(torch.load(generator_weights))
    stn.load_state_dict(torch.load(stn_weights))


    # inference on the patches
    # Note: this code just supports batch_size = 1 for now!
    DVF_dir = test_configs['save_dir'] + "exp" + str(experiment) + "_epoch" + str(epoch_idx) + "_DVF/"
    print("STEP1: Generating DVFs for case {} in {}".format(case_num, DVF_dir))
    ################################ 1. Generate DVF for patches ################################ 
    with torch.no_grad():
        for paired_patches, ID, info in tqdm(test_loader):

            # paired_patches shape: [batch, 2, d, h, w]   
            pair = paired_patches.to(device)
            DVF = FFCGenerator(pair).cpu()

            # save DVF    
            if not os.path.exists(DVF_dir):
                os.mkdir(DVF_dir)
            


            fi = torch.unsqueeze(pair[:, 0, :], 1).cpu() # fixed patch 
            mi = torch.unsqueeze(pair[:, 1, :], 1).cpu() # moving patch
            registered_images = stn(mi.permute(0, 1, 3, 4, 2), # (batch, c, h, w, d)
                                    DVF.permute(0, 1, 3, 4, 2))

            registered_images = registered_images.permute(0, 1, 4, 2, 3)

            # view3d_image(fi[0, 0, :, :, :], slice_axis=0)
            # view3d_image(mi[0, 0, :, :, :], slice_axis=0)
            # view3d_image(registered_images[0, 0, :, :, :], slice_axis=0)


            torch.save(DVF, DVF_dir + "DVF_case" + str(case_num) 
                                                 + "_f" + str(ID['fixed_image_phase_num'].item()) + "0"  # fixed image phase number
                                                 + "_m" + str(ID['moving_image_phase_num'].item()) + "0" # moving image phase number
                                                 + "_patch" + str(ID['patch_idx'].item()) + ".pt")

            del pair, DVF, fi, mi, registered_images

    
    
    print("STEP 2: Fusing image and DVF patches for case {}".format(case_num))
    ################################ 2. Patch fusion ################################
    # 2.1 fuse DVF patch images
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
    
    fused_dvf = fuse_dvf(dvfs, patch_size=(64, 64, 64), overlap=(8, 8, 8), output_size = resampled_dirlab_volume_sizes[case_num-1])

    print("STEP 3: Registration")
    ################################ 3. Registration ################################
    fixed_img = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(test_configs['test_data_path'] + 'case' + str(case_num) + '/case' + str(case_num) + '_T00_R.mha')))
    fixed_img = (fixed_img - torch.min(fixed_img)) / (torch.max(fixed_img) - torch.min(fixed_img))
    moving_img = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(test_configs['test_data_path'] + 'case' + str(case_num) + '/case' + str(case_num) + '_T50_R.mha')))
    moving_img = (moving_img - torch.min(moving_img)) / (torch.max(moving_img) - torch.min(moving_img))
    
    registered_img = stn(moving_img.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2), # (batch, c, w, h, d)
                                fused_dvf.unsqueeze(0).permute(0, 1, 4, 3, 2))

    registered_img = registered_img.permute(0, 1, 4, 3, 2)[0, 0, :, :, :]


    if save_figs:
        save_figs_dir = test_configs['save_dir'] + "exp" + experiment + "_epoch" + str(epoch_idx) + "_FIGS/"
        if not os.path.exists(save_figs_dir):
            os.mkdir(save_figs_dir)

        view3d_image(fixed_img, slice_axis=0, filename=save_figs_dir + "epoch" + str(epoch_idx) + "_case" + str(case_num) + "_T" + fixed_img_phase + "_fixed.png")
        view3d_image(moving_img, slice_axis=0, filename=save_figs_dir + "epoch" + str(epoch_idx) + "_case" + str(case_num) + "_T" + moving_img_phase + "_moving.png")
        view3d_image(registered_img, slice_axis=0, filename=save_figs_dir + "epoch" + str(epoch_idx) + "_case" + str(case_num) + "_registered.png")

        # save registered volume
        torch.save(registered_img, save_figs_dir + "epoch" + str(epoch_idx) + "_case" + str(case_num) + "_registered_volume.pt")

    # print("\nSTEP 4: TRE calculation")
    ################################ 4. TRE calculation ################################
    
    landmarks_root = "./data/DIRLAB/points/"
    
    fixed_img_landmark_path = landmarks_root + "case" + str(case_num) + "/case" + str(case_num) + "_300_T00_xyz_R.txt"
    moving_img_landmark_path = landmarks_root + "case" + str(case_num) + "/case" + str(case_num) + "_300_T50_xyz_R.txt"
    
    landmarks_array_fixed = np.loadtxt(fixed_img_landmark_path, dtype=float)
    landmarks_array_moving = np.loadtxt(moving_img_landmark_path, dtype=float)
    

    # 4.2 Compute TRE before/after registration
    # moving_img_info = info[1]
    # fused_dvf = fused_dvf * (moving_img_info['max_pixel'] - moving_img_info['min_pixel']) + moving_img_info['min_pixel']
    registered_landmarks_array, removed_indicies = create_registered_landmark(moving_img_landmark_path, fused_dvf)

    

    TRE_before_registration_mean, TRE_before_registration_std = compute_TRE(case_num, 
                                                                            landmarks_array_fixed, 
                                                                            landmarks_array_moving)
    print("TRE before registration for case {0} is: {1:.2f} +- {2:.2f}".format(case_num, TRE_before_registration_mean, TRE_before_registration_std))


    landmarks_array_fixed = np.delete(landmarks_array_fixed, removed_indicies, axis=0).copy()
    landmarks_array_moving = np.delete(landmarks_array_moving, removed_indicies, axis=0).copy()

    TRE_after_registration_mean, TRE_after_registration_std = compute_TRE(case_num, 
                                                                          registered_landmarks_array, 
                                                                          landmarks_array_fixed)

    
    print("TRE after registration for case {0} is: {1:.2f} +- {2:.2f}".format(case_num, TRE_after_registration_mean, TRE_after_registration_std))


    import ipdb; ipdb.set_trace()

for i in range(10):
    test_per_case("00", 60, case_num=i+1, save_figs=True)