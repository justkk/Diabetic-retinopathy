""" The utils module provides some basic pre-processing operations over fundus
    images.
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import img_as_bool, img_as_ubyte
from scipy.ndimage.morphology import binary_fill_holes
import ConfigParser


def parse_init_file(filename):
    """Read the initialization parameters file

    Parameters
    ----------
    filename : string
        path of ini file with initialization parameters

    Returns
    -------
    params : dictionary
        Dictionary of parameter values
    
    Example
    -------
    >>>params = parse_init_file('dridb.ini')

    """
    parser = ConfigParser.ConfigParser()
    parser.read(filename)
    params = dict(parser._sections)
    for k in params:
        params[k] = dict(parser._defaults, **params[k])
        params[k].pop('__name__', None)
    return params


def get_fundus_mask(img_in, threshold=None):
    """Obtain the Fundus mask of an image.

    This function takes a fundus image as the first argument and an optional
    threshold argument and computes the fundus mask of the input image.

    Parameters
    ----------
    img_in : Numpy array
        Input fundus image.
    threshold : float, optional
        Threshold for binarisation. If not provided,
        Otsu's threshold is computed and used. Default value is None.

    Returns
    -------
    mask : Numpy array.
        Binary fundus mask

    Example
    -------
    >>>mask = getfundusmask(img_in, threshold=10)

    """
    if len(img_in.shape) == 3:  # if the image is color take the green channel.
        img_in = img_in[:, :, 1]
    img_in = img_as_ubyte(img_in)
    if threshold is None:  # If the threshold is not provided, find the global otsu's threshold.
        threshold = threshold_otsu(img_in)
    mask = img_in > threshold  # Threshold the image.
    mask, num = label(mask, return_num=True)  # Obtain the label image.
    area = np.zeros(num)
    for i in range(num):
        area[i] = len(np.where(mask == i)[1])  # Regionprops is not needed.
        #  Area of each connected component is obtained in a simple loop.
    label_keep = np.argmax(area)
    mask[np.where(mask != label_keep)] = 0  # Keep only the component with the maximum area.
    mask = mask > 0  # Binarise the image.
    mask = binary_fill_holes(mask)  # Fill holes.
    mask = img_as_bool(mask)
    return mask
