'''
Program to pre process HR and LR image files for training. Given paths to HR and LR iamges the program preprocesses the
and saves the processed images at the specified directory.

Pre-process : divide the LR image into 25% overlapping pathches of size p_height x p_width.
divide the HR image into 25% overlapping pathches of sf*size p_height x sf*p_width.
'''

from __future__ import print_function, division
import numpy as np
import cv2
from skimage import io
from os import listdir
import skvideo.io

start = [0, 180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160, 2340, 2520, 2700, 2880, 3060, 3240,
         3420, 3600, 3780, 3960, 4140, 4320, 4500, 4680, 4860, 5040, 5220]
end = [240, 420, 600, 780, 960, 1140, 1320, 1500, 1680, 1860, 2040, 2220, 2400, 2580, 2760, 2940, 3120, 3300, 3480,
       3660, 3840, 4020, 4200, 4380, 4560, 4740, 4920, 5100, 5280, 5460]
sf = 2  # the factor by which to super-resolve images
h_index = 0  # start and end index for height of image
w_index = 0  # start and end index for width of image
p_height = 240  # patch height
p_width = 240  # patch width

# ===========================================================================================
# path to save processed HR files
f_save_hr = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/processed/part4_HR'
# path to save processed LR files
f_save_lr = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/processed/part4_LR'

# path to HR images
path_hr = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/child_part4_HR/*.png'
# path to LR images
path_lr = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/child_part4_LR/*.png'

# List names of all images in HR directory to arr_hr
arr_hr = listdir('C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/child_part4_HR')
# List names of all images in LR directory to arr_lr
arr_lr = listdir('C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/child_part4_LR')

# ===========================================================================================

pad_color = np.random.randint(2, 254, size=(len(arr), 3))


def load_data():
    """
    Loads data from the directory path_hr and path_lr
    :return: x, y
    """
    temp_x = io.imread_collection(path_lr)
    print("loaded_x")
    temp_y = io.imread_collection(path_hr)
    print("loaded_y")

    xx = [images for i, images in enumerate(temp_x)]
    yy = [images for i, images in enumerate(temp_y)]

    # Take only the first 3 channels, i.e. RGB
    xx = xx[0::3]
    yy = yy[0::3]

    x = np.array(xx)
    print("shape of x data", x.shape)

    y = np.array(yy)
    print("shape of y data", x.shape)

    print("data loaded")
    return x, y


# Function to set h_index and y_index
def util(x):
    _, height, width, channels = x.shape

    global h_index
    global w_index

    for j in range(len(start)):
        if start[j] < height:
            h_index = j
        if start[j] < width:
            w_index = j


# Function to preprocess data
def preprocess(x, y):
    util(x)

    # PRE PROCESS X
    _, height, width, channels = x.shape

    # flag to check if pre processing is required or not -- rudundant, not actually required
    pre_flag = 1

    # iph_app = height of padding to be appended to the LR i/p image
    # oph_app = height of padding to be appended to the HR i/p image
    iph_app = end[h_index] - height
    oph_app = (sf * iph_app)

    # ipw_app = width of padding to be appended to the LR i/p image
    # opw_app = width of padding to be appended to the HR i/p image
    ipw_app = end[w_index] - width
    opw_app = (sf * ipw_app)

    if pre_flag == 1:
        for k in range(x.shape[0]):
            # pad image with appropritate border to make it suitable for pre procesing
            gg = cv2.copyMakeBorder(x[k], top=0, bottom=iph_app, left=0, right=ipw_app, borderType=cv2.BORDER_CONSTANT,
                                    value=(int(pad_color[k][0]), int(pad_color[k][1]), int(pad_color[k][2])))
            z = 0
            for i in range(h_index + 1):
                for j in range(w_index + 1):
                    z = z + 1
                    gx = gg[start[i]: end[i], start[j]: end[j], :]
                    gx = np.uint8(gx)

                    name = arr_hr[k]
                    name = f_save_lr + '/pro_' + name[:-4] + '_{0}'.format(z) + name[-4:]
                    skvideo.io.vwrite(name, gx)

        print('x data prepared', k, z)


    # PRE PROCESS Y
    _, height, width, channels = y.shape

    if pre_flag == 1:
        # y.shape[0]
        for k in range(y.shape[0]):
            # pad image with appropritate border to make it suitable for pre procesing
            gg = cv2.copyMakeBorder(y[k], top=0, bottom=oph_app, left=0, right=opw_app, borderType=cv2.BORDER_CONSTANT,
                                    value=(int(pad_color[k][0]), int(pad_color[k][1]), int(pad_color[k][2])))
            z = 0
            for i in range(h_index + 1):
                for j in range(w_index + 1):
                    z = z + 1
                    gy = gg[(sf * start[i]): (sf * end[i]), (sf * start[j]): (sf * end[j]), :]
                    gy = np.uint8(gy)

                    name = arr_lr[k]
                    name = f_save_hr + '/pro_' + name[:-4] + '_{0}'.format(z) + name[-4:]
                    skvideo.io.vwrite(name, gy)

        print('y data prepared', k, z)

        print("\nPre processing done.\n")


if __name__ == "__main__":
    x, y = load_data()
    preprocess(x, y)
