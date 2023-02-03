import pydicom
import os
import glob
import numpy as np
import SimpleITK as sitk
import utils.image_processing as ip
import matplotlib.pyplot as plt

# dcm_path = "./data/Nabavi/Patient-1/00/00-slice001.dcm"
# dicom = pydicom.read_file(dcm_path)
# array = dicom.pixel_array
#
# print(array.shape)


def write_volumes(root):
    for case in sorted(os.listdir(root))[-2:]:
        # case_directory: Case1Pack, Case2Pack, ...

        if not case.startswith("Case"):
            continue
        case_num = case[4]

        case_dir = os.path.join(root, case)
        for phase_number in sorted(os.listdir(case_dir)):
            if phase_number.startswith(".DS"):
                continue
            phase_dir = os.path.join(case_dir, phase_number)
            # print(phase_number)
            number_of_slices = len(glob.glob(os.path.join(phase_dir, "*.dcm")))
            numpy_volume = np.empty((number_of_slices, 512, 512))
            slice_index = 0
            for dcm_path in sorted(glob.glob(os.path.join(phase_dir, "*.dcm"))):
                dicom = pydicom.read_file(dcm_path)
                # now its time to add the slicwhatse to the volume
                numpy_volume[slice_index, :, :] = dicom.pixel_array
                slice_index += 1

            mha_dir = os.path.join(root, "mha")
            if not os.path.isdir(mha_dir):
                os.mkdir(mha_dir)

            where_to_write = os.path.join(mha_dir, "case" + case_num)
            if not os.path.isdir(where_to_write):
                os.mkdir(where_to_write)

            # print(numpy_volume.shape)
            image_sitk = ip.array_to_sitk(numpy_volume)
            sitk.WriteImage(image_sitk, where_to_write + "/case" + case_num + "_T" + phase_number + ".mha")
            print('case' + case_num + ' phase' + phase_number + " with shape " + str(numpy_volume.shape) + ' is done..')


if __name__ == '__main__':
    # write_volumes(root="../data/SBU/")
    dicom = pydicom.read_file("../data/SBU/xx/1-055.dcm")
    np_array = dicom.pixel_array
    print(np_array.shape)
    plt.imshow(np_array, cmap="gray")
    plt.show()