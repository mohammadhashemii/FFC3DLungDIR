import numpy as np

original_dirlab_volume_spacing = [[0.97, 0.97, 2.5],
                                    [1.16, 1.16, 2.5],
                                    [1.15, 1.15, 2.5],
                                    [1.13, 1.13, 2.5],
                                    [1.10, 1.10, 2.5],
                                    [0.97, 0.97, 2.5],
                                    [0.97, 0.97, 2.5],
                                    [0.97, 0.97, 2.5],
                                    [0.97, 0.97, 2.5],
                                    [0.97, 0.97, 2.5]]


def create_registered_landmark(landmark_txt_file_adr, dvf):

    # dvf shape: (3, z, x, y)
    # landmark shape: (300, x, y, z)
    landmarks_array = np.loadtxt(landmark_txt_file_adr, dtype=np.float32)
    landmarks_array_transformed = landmarks_array.copy()

    removed_indicies = []
    for i in range(len(landmarks_array)):
        l_x, l_y, l_z = int(landmarks_array[i, 0]), int(landmarks_array[i, 1]), int(landmarks_array[i, 2])
        if l_x < dvf.shape[2] and l_y < dvf.shape[3] and l_z < dvf.shape[1]:

            dx = dvf[2, l_z, l_y, l_x].item()
            
            dy = dvf[1, l_z, l_y, l_x].item()
            dz = dvf[0, l_z, l_y, l_x].item()

            # print(dx, dy, dz)

            landmarks_array_transformed[i, 0] = landmarks_array[i, 0] + dx  # x
            landmarks_array_transformed[i, 1] = landmarks_array[i, 1] + dy # y
            landmarks_array_transformed[i, 2] = landmarks_array[i, 2] + dz  # z
        else:
            removed_indicies.append(i)

    print("{} of landmarks are not considered due to patch extraction!".format(len(removed_indicies)))
    # landmarks_array = np.delete(landmarks_array, removed_indicies, axis=0).copy()

    # print(landmarks_array)

    return landmarks_array_transformed, removed_indicies

def compute_TRE(case_num, landmarks_array_fixed, landmarks_array_moving):

    l1 = landmarks_array_fixed.copy()
    l2 = landmarks_array_moving.copy()

    # l1[:, 0] *= original_dirlab_volume_spacing[case_num-1][0]
    # l1[:, 1] *= original_dirlab_volume_spacing[case_num-1][1]
    # l1[:, 2] *= original_dirlab_volume_spacing[case_num-1][2]

    # l2[:, 0] *= original_dirlab_volume_spacing[case_num-1][0]
    # l2[:, 1] *= original_dirlab_volume_spacing[case_num-1][1]
    # l2[:, 2] *= original_dirlab_volume_spacing[case_num-1][2]

    norm = [np.linalg.norm(l1[i] - l2[i]) for i in range(len(l2))]

    # print("TRE: {} +- {}".format(np.mean(norm), np.std(norm)))
    return np.mean(norm), np.std(norm)