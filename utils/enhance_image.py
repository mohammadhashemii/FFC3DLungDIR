import SimpleITK as sitk
import skimage

# input image address, write image to original image
# the smaller the first number in the scale range
# each entry in the mask has range[0, 1], compute as enhanced = original + mask_weight * mask
def enhance_image(image_address, _scale_range = (0.51, 2), mask_weight = 2000): 
    image = sitk.ReadImage(image_address)
    image_array = sitk.GetArrayFromImage(image)
    mask = skimage.filters.frangi(image_array, black_ridges=False, beta=0.5, scale_range=_scale_range, scale_step=0.5)
    enhanced_image_array = image_array + mask * mask_weight
    enhanced_image = sitk.GetImageFromArray(enhanced_image_array)
    new_address = image_address.split(".mha")[0] + "_enhanced.mha"
    sitk.WriteImage(enhanced_image, new_address)
    return 

if __name__ == '__main__':
    creatis_dir = './data/CREATIS/'
    to_be_enhanced_list = [0, 1, 2, 3, 4, 5]
    for e in to_be_enhanced_list:
        for p in range(10):
            resampled_img_address = creatis_dir + str(e) + '/' + str(p) + '0_R.mha'
            enhance_image(resampled_img_address)
    
