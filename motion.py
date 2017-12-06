from skimage.transform import rotate
from skimage import img_as_ubyte, img_as_float
from skimage import exposure
import numpy as np
from random import randint
def get_rotational_gmp(img_in, coalesce_type="MAX", angle_min=-5, angle_step=1, angle_max=6, pivot=(-1,-1)):
    """This function creates a rotational GMP of an input image.

       get_gmp takes an input image alongwith a set of parameters and
       computes the rotational GMP and returns it.

       Parameters
       -----------
       img_in : numpy array
                img_in is the input image
       coalesce_type : string
                Specifies the coalescing function for gmp.
                Accepts, "MAX", "MIN" and "MEAN". Defaul is "MAX"
       angle_min : float
                Specifies the minimum rotation angle in degrees.
                Default is -5.
       angle_step : float
                Specifies the step angle in degrees for gmp. 
                Default is 1.
       angle_max : float
                Specifies the maximum angle in degrees for gmp.
                Default is 1.
       pivot : tuple
              Specifies the center of rotation. Default is (-1,-1)
              meaning the center of the image.

     Returns
     --------
      gmp_out : numpy array.
                Output GMP image.

     Examples
     ---------
      >>>gmp = get_gmp(img_in, coalesce_type="MEAN", angle_min=-5, angle_step=1, angle_max=5, pivot=(25,55))
    """
    if len(img_in.shape)==3:  #If the image is color, take green channel.
        img_in = img_in[:,:,2]
    img_in = img_as_float(img_in)
    gmp_out = np.zeros(img_in.shape)
    temp_img = np.zeros(img_in.shape)
    if pivot == (-1,-1):
        pivot = (np.ceil(img_in.shape[0]/2), np.ceil(img_in.shape[1]/2))
    counter = 0
    for i in range(angle_min, angle_max, angle_step):
        temp_img = rotate(img_in, i, center=pivot)
        counter = counter + 1
        if not gmp_out.any():
            gmp_out = temp_img
        elif coalesce_type is "MAX":
            gmp_out = np.maximum.reduce([gmp_out, temp_img])
        elif coalesce_type is "MIN":
            gmp_out = np.minimum.reduce([gmp_out,temp_img])
        elif coalesce_type is "MEAN":
            gmp_out = (gmp_out * (counter - 1) + temp_img)/counter
    gmp_out = img_as_ubyte(gmp_out)
    return gmp_out

def get_interference_variance(img_in, num_pivots=100, coalesce_type="MAX", angle_min=-5, angle_step=1, angle_max=5):
    """ This function returns the interference and variance maps
    
        This function takes a single channel image and computes the interference 
        and variance maps based on parameters provided.

        Parameters
        -----------
        img_in : Numpy array
                 Input image. Must be single channel.
        num_pivots : integer.
                     Number of pivot points (Randomly chosen).
                     Default value is 100.
        coalesce_type : string.
                     Coalescing function used to create the GMP.
                     Default value is "MAX"
        angle_min : float
                    Minimum angle for rotation in degrees.
                    Default value is -5.
        angle_step : float
                    Step angle for rotation in degrees.
                    Default value is 1.
        angle_max : float
                    Maximum angle for rotation in degrees.
                    Default value is 5.
        
        Returns
        --------
        A Dictionary with following keys:
         'interferencemap' :  Computed interference map, ubyte.
         'variancemap' :  Computed variance map, ubyte.
       """
       
    if len(img_in.shape)==3:
        img_in = img_in[:,:,2]
    int_out = np.zeros(img_in.shape)
    var_out = np.zeros(img_in.shape + (num_pivots,))
    img_in = img_as_float(img_in)
    for i in range(1,num_pivots+1):
        pivot = (randint(0,img_in.shape[0]-1), randint(0,img_in.shape[1]-1))
        temp_gmp = get_rotational_gmp(img_in, coalesce_type, angle_min, angle_step, angle_max, pivot)
        int_out = int_out + img_as_float(temp_gmp)
        var_out[:,:,i-1] = img_as_float(temp_gmp)
    int_out = exposure.rescale_intensity(int_out)
    int_out = img_as_ubyte(int_out)
    var_out = np.var(var_out, axis=2)
    var_out = exposure.rescale_intensity(var_out)
    var_out = img_as_ubyte(var_out)
    return {'interferencemap':int_out, 'variancemap':var_out}
    

        
        
        
            
        

    
