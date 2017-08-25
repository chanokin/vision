from ..common import *

def cs_3x3():
    s2 = 1/np.sqrt(2)
    c = 4.*( 1 + s2 )
    cs = np.array([[-s2, -1, -s2], [-1, c, -1], [-s2, -1, -s2]])
    
    return conv2one( cs )


def center_surround_kernel(width=3, ctr_sigma=0.8, sigma_mult=6.7, on_center=True):
    '''Compute the convolution kernel for center-surround behaviour,
       it sums to 0 and self-convolves to 1
       :param width: matrix width (we use square matrices)
       :param ctr_sigma: standard deviation for the center component
       :param sigma_mult: sigma_mult*ctr_sigma is the standard deviation 
                          for the surround component of the kernel

       :return cs_kernel: center-surround kernel
    '''
    
    cs_kernel = None
    if width == 3:
        return cs_3x3()[0]
    else:
        ctr = gaussian2D(width, ctr_sigma)
        srr = gaussian2D(width, sigma_mult*ctr_sigma)
        if on_center:
            cs_kernel = ctr - srr
        else:
            cs_kernel = srr - ctr

        cs_kernel = sum2zero(cs_kernel)
        cs_kernel = conv2one(cs_kernel)[0]
    
    return cs_kernel


def gaussian2D(width, sigma_x, sigma_y=None, theta=0, step=1):
    '''Create a matrix with values that follow a 2D Gaussian
       function. 
       :param width: width of matrix
       :param sigma: standard deviation for the Gaussian

       :return gauss: 2D Gaussian
    '''
    if sigma_y is None:
        sigma_y = sigma_x

    theta = np.deg2rad(theta)
    a = 0.5 * ((np.cos(theta) / sigma_x) ** 2 + (np.sin(theta) / sigma_y) ** 2)
    b = 0.25 * (np.sin(2. * theta) * (-1. / (sigma_x ** 2) + 1. / (sigma_y ** 2)))
    c = 0.5 * ((np.sin(theta) / sigma_x) ** 2 + (np.cos(theta) / sigma_y) ** 2)

    half_width = width // 2
    x, y = np.meshgrid(np.arange(-half_width, half_width + step, step),
                       np.arange(-half_width, half_width + step, step))

    gauss = np.exp(-(a * (x ** 2) + 2. * b * x * y + c * (y ** 2)))
    gauss /= np.sum(gauss)
    return gauss


def correlationGaussian2D(width0, width1, sigma_x0, sigma_x1,
                          sigma_y0=None, sigma_y1=None,
                          theta0=0, theta1=0,
                          use_first_width=True,
                          step=1):
    '''Create a matrix with values that follow the correlation of two
       2D Gaussian functions. 
       :param width: width of matrix
       :param sigma: standard deviation for the Gaussian

       :return gauss: 2D Gaussian
    '''
    if use_first_width:
        width = width0
    else:
        width = max(width0, width1)

    half_width = width // 2
    sigma_2 = (sigma_x0 ** 2 + sigma_x1 ** 2)
    x, y = np.meshgrid(np.arange(-half_width, half_width + 1),
                       np.arange(-half_width, half_width + 1))
    x_2_plus_y_2 = x ** 2 + y ** 2

    norm_weight = (1. / (2. * np.pi * sigma_2))
    gauss = (norm_weight * np.exp((-x_2_plus_y_2) / (2. * sigma_2)))

    return gauss


def split_center_surround_kernel(width=3, ctr_sigma=0.8, sigma_mult=6.7):
    
    cs_kernel = None
    if width == 3:
        cs_kernel, w = cs_3x3()
        s2 = 1/np.sqrt(2)
        c = 4.*( 1 + s2 )
        ctr = np.array([[0, 0, 0], [0, c, 0], [0, 0, 0]])
        srr = np.array([[s2, 1, s2], [1, c, 1], [s2, 1, s2]])
    else:
        ctr = gaussian2D(width, ctr_sigma)
        ctr, w = normalize(ctr)
        
        srr = gaussian2D(width, sigma_mult*ctr_sigma)
        srr, w = normalize(srr)
        
        cs_kernel = ctr - srr
        cs_kernel, w = conv2one(cs_kernel)
        
        ctr *= w
        srr *= w
        srr = -srr
        
    return [ctr, srr] #excitatory first, inhibitory later

