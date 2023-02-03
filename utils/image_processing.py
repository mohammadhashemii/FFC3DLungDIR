import SimpleITK as sitk
import math

def array_to_sitk(array_input, origin=None, spacing=None, direction=None, is_vector=False, im_ref=None):
    if origin is None:
        origin = [0, 0, 0]
    if spacing is None:
        spacing = [1, 1, 1]
    if direction is None:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    sitk_output = sitk.GetImageFromArray(array_input, isVector=is_vector)
    if im_ref is None:
        sitk_output.SetOrigin(origin)
        sitk_output.SetSpacing(spacing)
        sitk_output.SetDirection(direction)
    else:
        sitk_output.SetOrigin(im_ref.GetOrigin())
        sitk_output.SetSpacing(im_ref.GetSpacing())
        sitk_output.SetDirection(im_ref.GetDirection())
    return sitk_output



def index_to_world(landmark_index, spacing=None, origin=None, direction=None, im_ref=None):
    if im_ref is None:
        if spacing is None:
            spacing = [1, 1, 1]
        if origin is None:
            origin = [0, 0, 0]
        if direction is None:
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    else:
        spacing = list(im_ref.GetSpacing())
        origin = list(im_ref.GetOrigin())
        direction = list(im_ref.GetDirection())
    landmarks_point = [None] * len(landmark_index)
    for p in range(len(landmark_index)):
        landmarks_point[p] = [index * spacing[i] + origin[i] for i, index in enumerate(landmark_index[p])]
    return landmarks_point


def world_to_index(landmark_point, spacing=None, origin=None, direction=None, im_ref=None):
    if im_ref is None:
        if spacing is None:
            spacing = [1, 1, 1]
        if origin is None:
            origin = [0, 0, 0]
        if direction is None:
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    else:
        spacing = list(im_ref.GetSpacing())
        origin = list(im_ref.GetOrigin())
        direction = list(im_ref.GetDirection())
    landmarks_index = [None] * len(landmark_point)
    for p in range(len(landmark_point)):
        landmarks_index[p] = [round(point - origin[i] / spacing[i])  for i, point in enumerate(landmark_point[p])]
    return landmarks_index

def resampler_by_transform(im_sitk, dvf_t, im_ref=None, default_pixel_value=0, interpolator=sitk.sitkBSpline):
    if im_ref is None:
        im_ref = sitk.Image(dvf_t.GetDisplacementField().GetSize(), sitk.sitkInt8)
        im_ref.SetOrigin(dvf_t.GetDisplacementField().GetOrigin())
        im_ref.SetSpacing(dvf_t.GetDisplacementField().GetSpacing())
        im_ref.SetDirection(dvf_t.GetDisplacementField().GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(dvf_t)
    out_im = resampler.Execute(im_sitk)
    return out_im


def resampler_sitk(image_sitk, spacing=None, scale=None, im_ref=None, im_ref_size=None, default_pixel_value=0, interpolator=sitk.sitkBSpline, dimension=3):
    """
    :param image_sitk: input image
    :param spacing: desired spacing to set
    :param scale: if greater than 1 means downsampling, less than 1 means upsampling
    :param im_ref: if im_ref available, the spacing will be overwritten by the im_ref.GetSpacing()
    :param im_ref_size: in sikt order: x, y, z
    :param default_pixel_value:
    :param interpolator:
    :param dimension:
    :return:
    """
    if spacing is None and scale is None:
        raise ValueError('spacing and scale cannot be both None')

    if spacing is None:
        spacing = tuple(i * scale for i in image_sitk.GetSpacing())
        if im_ref_size is None:
            im_ref_size = tuple(round(i / scale) for i in image_sitk.GetSize())

    elif scale is None:
        ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
        if im_ref_size is None:
            im_ref_size = tuple(math.ceil(size_dim * ratio[i]) - 1 for i, size_dim in enumerate(image_sitk.GetSize()))
    else:
        raise ValueError('spacing and scale cannot both have values')

    if im_ref is None:
        im_ref = sitk.Image(im_ref_size, sitk.sitkInt8)
        im_ref.SetOrigin(image_sitk.GetOrigin())
        im_ref.SetDirection(image_sitk.GetDirection())
        im_ref.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref,
                                            default_pixel_value=default_pixel_value,
                                            interpolator=interpolator)
    return resampled_sitk