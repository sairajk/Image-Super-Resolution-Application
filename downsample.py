'''
A python program to downsample images.
This saves the resized images with same name and extension as input image in the destination folder
'''

from __future__ import print_function, division, absolute_import
import numpy as np
import cv2
from skimage import io
from os import listdir

# ============================================================================
# Destination to save downsampled images
f_save = "C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/ink2"

# Destination of input files
f_inp = "C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/ink"

# scaling factor for image in x and y direction, assign to "None",
# if you want to scale images by exact sizes
fx = 0.5
fy = 0.5

# the size of the image in x direction and y direction, used only if fx and fy are None.
x = None
y = None
# =============================================================================

arr = listdir(f_inp)


def load_data():
    temp_y = io.imread_collection(f_inp + "*.png")
    y = np.array([images for i, images in enumerate(temp_y)])

    print("Loaded High Resolution images")
    return y


def downsample(data):
    for i in range(data.shape[0]):
        img = data[i]

        if fx is None:
            small = cv2.resize(img, (x, y))
        else:
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        name = arr[i]
        path = f_save + "/" + name
        cv2.imwrite(path, small)


if __name__ == "__main__":
    data = load_data()
    downsample(data)

    print("Resizing done")
