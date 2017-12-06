from utils import *
from motion import *
from skimage import io
import matplotlib.pyplot as plt
if __name__ == '__main__':
    #test code for ini file parser
    init_file = 'dridb.ini'
    params = parse_init_file(init_file)
    print 'read init file'
    print params
    #test code for get_fundus_mask()
    img_in = io.imread('./data/assorted-samples/fundus_img1.jpg')
    mask = get_fundus_mask(img_in, threshold=10)  #Test with optional argument.
    mask = get_fundus_mask(img_in)  #Test without optional.
    print "get_fundus_mask() successfully tested."
    #test code for get_gmp()
    gmp = get_rotational_gmp(img_in, coalesce_type="MAX", angle_min=0, angle_step=1, angle_max=180)
    io.imshow(gmp)
    plt.show()
    print "get_gmp() successfully tested."
    #test code for get_int_var()
    int_var_dict = get_interference_variance(img_in, num_pivots=10, coalesce_type="MAX", angle_min=-5, angle_step=1, angle_max=6)
    io.imshow(int_var_dict['interferencemap'])
    plt.show()
    io.imshow(int_var_dict['variancemap'])
    plt.show()
    print "get_int_var() successfully tested."
