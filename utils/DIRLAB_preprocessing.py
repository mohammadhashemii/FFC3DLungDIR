# https://github.com/hsokooti/RegNet/blob/28a8b6132677bb58e9fc811c0dd15d78913c7e86/functions/preprocessing/resampling_isotropic.py

import SimpleITK as sitk
import image_processing as ip

DEFAULT_PIXEL_VALUE = -1024  # for DIRLAB dataset: the pixel value when transformed pixel is outside of the image

def resampling(raw_img_address, spacing=None, requested_im_list=None):
    if spacing is None:
        spacing = [1, 1, 1]

    if requested_im_list is None:
        requested_im = 'Im'


    if requested_im == 'Im':
        interpolator = sitk.sitkBSpline
    elif requested_im in ['Lung', 'Torso']:
        interpolator = sitk.sitkNearestNeighbor
    
    im_raw_sitk = sitk.ReadImage(raw_img_address)

    im_resampled_sitk = ip.resampler_sitk(im_raw_sitk,
                                            spacing=spacing,
                                            default_pixel_value=DEFAULT_PIXEL_VALUE,
                                            interpolator=interpolator,
                                            dimension=3)
    
    filename = '.'.join(raw_img_address.split('.')[:-1]) + '_R.mha' 
    sitk.WriteImage(im_resampled_sitk, filename)


def crop_image(raw_img_address, X_start, width, Y_start, height):
    im_raw_sitk = sitk.ReadImage(raw_img_address)
    image_array = sitk.GetArrayFromImage(im_raw_sitk)

    cropped_sitk_image = sitk.GetImageFromArray(image_array[:, X_start:X_start+width, Y_start:Y_start+height])
    sitk.WriteImage(cropped_sitk_image, raw_img_address)


if __name__ == '__main__':
    dirlab_dir = './data/DIRLAB/mha/'
    # we first need to do isotropic resampling
    case_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for c in case_list:
        for p in range(10):
            raw_img_address = dirlab_dir + 'case'+ str(c) + '/case' + str(c) + '_T' + str(p) + '0.mha'

            resampling(raw_img_address, spacing=[1, 1, 1])


    
    # cropping two of the cases : case 4 and 5
    # to_be_cropped_case_list = [4, 5]

    # for c in to_be_cropped_case_list:
    #     for p in range(10):

    #         resampled_img_address = dirlab_dir + str(c) + '/' + str(p) + '0_R.mha'
    #         crop_image(resampled_img_address, 100, 400, 100, 400)
