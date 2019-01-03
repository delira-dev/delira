from scipy.ndimage import zoom
import SimpleITK as sitk
import numpy as np
from .decorators import dtype_func

sitk_img_func = dtype_func(sitk.Image)


def calculate_origin_offset(new_spacing, old_spacing):
    """
    Calculates the origin offset of two spacings

    Parameters
    ----------
    new_spacing : list or np.ndarray or tuple
        new spacing
    old_spacing : list or np.ndarray or tuple
        old spacing

    Returns
    -------
    np.ndarray
        origin offset
    """
    return np.subtract(new_spacing, old_spacing)/2


@sitk_img_func
def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0),
                             interpolator=sitk.sitkLinear,
                             default_value=0.):
    """
    Resamples SITK Image to a given spacing

    Parameters
    ----------
    image : SimpleITK.Image
        image which should be resampled
    new_spacing : list or np.ndarray or tuple
        target spacing
    interpolator : Any
        implements the actual interpolation
    default_value : float
        default value

    Returns
    -------
    SimpleITK.Image
        resampled Image with target spacing

    """
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(
        zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size,
                                           spacing=new_spacing,
                                           direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset,
                                           default_value=default_value)
    return sitk_resample_to_image(image, reference_image,
                                  interpolator=interpolator,
                                  default_value=default_value)


@sitk_img_func
def sitk_resample_to_image(image, reference_image, default_value=0.,
                           interpolator=sitk.sitkLinear, transform=None,
                           output_pixel_type=None):
    """
    Resamples Image to reference image

    Parameters
    ----------
    image : SimpleITK.Image
        the image which should be resampled
    reference_image : SimpleITK.Image
        the resampling target
    default_value : float
        default value
    interpolator : Any
        implements the actual interpolation
    transform : Any (default: None)
        transformation
    output_pixel_type : Any (default:None)
        type of output pixels

    Returns
    -------
    SimpleITK.Image
        resampled image

    """
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    """
    Create a new blank image with given properties

    Parameters
    ----------
    size : list or np.ndarray or tuple
        new image size
    spacing : list or np.ndarray or tuple
        spacing of new image
    direction :
        new image's direction
    origin :
        new image's origin
    default_value : float
        new image's default value

    Returns
    -------
    SimpleITK.Image
        Blank image with given properties

    """
    image = sitk.GetImageFromArray(
        np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


@sitk_img_func
def sitk_resample_to_shape(img, x, y, z, order=1):
    """
    Resamples Image to given shape

    Parameters
    ----------
    img : SimpleITK.Image
    x : int
        shape in x-direction
    y : int
        shape in y-direction
    z : int
        shape in z-direction
    order : int
        interpolation order

    Returns
    -------
    SimpleITK.Image
        Resampled Image

    """
    img_np = sitk.GetArrayFromImage(img)
    img_np_fixed_size = zoom(img_np,
                             [z / img_np.shape[0],
                              y / img_np.shape[1],
                              x / img_np.shape[2]],
                             order=order)
    return sitk.GetImageFromArray(img_np_fixed_size)


@sitk_img_func
def max_energy_slice(img):
    """Determine the axial slice in which the image energy is max

    Parameters
    ----------
    img : SimpleITK.Image
        given image

    Returns
    -------
    int
        slice index
    """
    assert img.GetDimension() == 3
    return int(np.argmax(np.sum(sitk.GetArrayFromImage(img), axis=(1, 2))))


def sitk_copy_metadata(img_source, img_target):
    """ Copy metadata (=DICOM Tags) from one image to another

    Parameters
    ----------
    img_source : SimpleITK.Image
        Source image
    img_target : SimpleITK.Image
        Source image

    Returns
    -------
    SimpleITK.Image
        Target image with copied metadata
    """
    for k in img_source.GetMetaDataKeys():
        img_target.SetMetaData(k, img_source.GetMetaData(k))
    return img_target


@sitk_img_func
def bounding_box(mask, margin=None):
    """Calculate bounding box coordinates of binary mask

    Parameters
    ----------
    mask : SimpleITK.Image
        Binary mask
    margin : int, default: None
        margin to be added to min/max on each dimension

    Returns
    -------
    tuple
        bounding box coordinates of the form (xmin, xmax, ymin, ymax,
        zmin, zmax)
    """
    # mask_arr is in z, y, x order
    mask_arr = sitk.GetArrayFromImage(mask)
    zdim, ydim, xdim = mask_arr.shape
    nz = np.where(mask_arr != 0)
    lower = [np.min(nz[0]), np.min(nz[1]), np.min(nz[2])]
    upper = [np.max(nz[0]), np.max(nz[1]), np.max(nz[2])]
    if margin is not None:
        for axis in range(3):
            # make sure lower bound with margin is valid
            if lower[axis] - margin >= 0:
                lower[axis] -= margin
            else:
                lower[axis] = 0
            # make sure upper bound with margin is valid
            if upper[axis] + margin <= mask_arr.shape[axis] - 1:
                upper[axis] += margin
            else:
                upper[axis] = mask_arr.shape[axis] - 1
    bbox = lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]
    return bbox
