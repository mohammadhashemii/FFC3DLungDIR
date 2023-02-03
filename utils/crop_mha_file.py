import SimpleITK as ITK
import numpy as np
import image_processing as ip

# dirlab_info['case1']['Spacing'] = [0.97, 0.97, 2.5]
# dirlab_info['case2']['Spacing'] = [1.16, 1.16, 2.5]
# dirlab_info['case3']['Spacing'] = [1.15, 1.15, 2.5]
# dirlab_info['case4']['Spacing'] = [1.13, 1.13, 2.5]
# dirlab_info['case5']['Spacing'] = [1.10, 1.10, 2.5]
# dirlab_info['case6']['Spacing'] = [0.97, 0.97, 2.5]
# dirlab_info['case7']['Spacing'] = [0.97, 0.97, 2.5]
# dirlab_info['case8']['Spacing'] = [0.97, 0.97, 2.5]
# dirlab_info['case9']['Spacing'] = [0.97, 0.97, 2.5]
# dirlab_info['case10']['Spacing'] = [0.97, 0.97, 2.5]

for i in range(10):
    im_img_address = "./data/DIR-Lab/4DCT/mha2/case10/case10_T" + str(i) + "0.mha"
    im_imh_address = "./data/DIR-Lab/4DCT/mha_cropped/case10/case10_T" + str(i) + "0.mha"
    if i == 0 or i == 5:
        old_landmark_address = "./data/DIR-Lab/4DCT/points/case10/case10_300_T" + str(i) + "0_xyz_tr.txt"
        new_landmark_address = "./data/DIR-Lab/4DCT/points_cropped/case10/case10_300_T" + str(i) + "0_xyz_tr.txt"
        landmarks = np.loadtxt(old_landmark_address, dtype=int)
        landmarks[:, 0] =  landmarks[:, 0] - 100 # X
        landmarks[:, 1] =  landmarks[:, 1] - 80 # Y

        np.savetxt(new_landmark_address, landmarks, fmt='%.0f')

    mha = ITK.ReadImage(im_img_address)
    img = ITK.GetArrayFromImage(mha)
    im_data = np.array(img, np.int16)
    im_data =  im_data[:, 80:380, 100:400]
    print(im_data.shape)
    image_sitk = ip.array_to_sitk(im_data, spacing= [0.97, 0.97, 2.5], origin=[0, 0, 0])
    ITK.WriteImage(image_sitk, im_imh_address)