import pdb
import cv2
import numpy as np


# Settings
# These settings can be updated when importing this module e.g.
# import superpixels as sp
# sp.num_superpixels = 3000
num_superpixels = 4000
num_levels = 4
prior = 2
num_histogram_bins = 7
num_iterations = 520

def extract_superpixels(image):
    """ Extract superpixels from an image
    """
    global num_superpixels, num_levels, prior, num_histogram_bins, num_iterations
    
    height,width,channels = image.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
                                               num_superpixels, num_levels, prior, num_histogram_bins)
    color_img = np.zeros((height,width,3), np.uint8)
    color_img[:] = (0, 0, 255)
    seeds.iterate(image, num_iterations)
    maps = seeds.getLabels()
    return maps


def draw_superpixels(image, labels):
    # labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(False)

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)

    return result