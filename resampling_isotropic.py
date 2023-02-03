import settings as su
import SimpleITK as sitk
import utils.image_processing as ip
import os


def resampling_setting():
    resampling(data='DIR-Lab', spacing=[1, 1, 1], requested_im_list=['Im'])
    # resampling(data='SBU', spacing=[1, 1, 1], requested_im_list=['Im'])


def resampling(data, spacing=None, requested_im_list=None):
    if spacing is None:
        spacing = [1, 1, 1]
    if requested_im_list is None:
        requested_im_list = ['Im']
    setting = su.initialize_setting('')
    print(setting)

    for phase_im in range(len(setting['data'][data]['phases'])):
        for cn in setting['data'][data]['case_numbers_list']:
            im_info_su = {'data': data, 'phase': phase_im, 'cn': cn}
            for requested_im in requested_im_list:
                if requested_im == 'Im':
                    interpolator = sitk.sitkBSpline
                elif requested_im in ['Lung', 'Torso']:
                    interpolator = sitk.sitkNearestNeighbor
                else:
                    raise ValueError('interpolator is only defined for ["Im", "Mask", "Torso"] not for '+requested_im)
                img_path = os.path.join(setting['data'][data]['root'], data, "4DCT", "mha_cropped", "case" + str(im_info_su['cn']),
                                        "case" + str(cn) + "_" + setting['data'][data]['phases'][phase_im] + setting['data'][data]['ext'])
                im_raw_sitk = sitk.ReadImage(img_path)
                im_resampled_sitk = ip.resampler_sitk(im_raw_sitk,
                                                      spacing=spacing,
                                                      default_pixel_value=setting['data'][data]['default_pixel_value'],
                                                      interpolator=interpolator,
                                                      dimension=3)
                resampled_img_path = os.path.join(setting['data'][data]['root'], data, "4DCT", "mha_cropped", "case" + str(im_info_su['cn']),
                                        "case" + str(cn) + "_" + setting['data'][data]['phases'][phase_im] + "_RS1" + setting['data'][data]['ext'])
                sitk.WriteImage(im_resampled_sitk, resampled_img_path)


if __name__ == '__main__':
    resampling_setting()
