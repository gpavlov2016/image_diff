import superpixels as sp
import cv2
import imageio
import numpy as np
import pdb
from sklearn.mixture import GaussianMixture

def run():
    filename = './data/toothpick.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    # List of superpixels
    masks = []

    # List of Edge Adjacency Matrix
    W = []

    # List of ID of superpixel that overlaps with the fixation point
    obj_ids = []

    for i in range(1):
        image = get_smaller_image(vid, i)
        sp.num_superpixels = 100
        mask = sp.extract_superpixels(image)
        masks.append(mask)

        # fixation point is assumed to be at the center of the image.
        fp = mask[int(image.shape[0]/2.0), int(image.shape[1]/2.0)]
        obj_ids.append(fp)
    
    # Learn background color model
    clf_o = GaussianMixture(n_components=1)
    clf_b = GaussianMixture(n_components=1)
    for m, mask in enumerate(masks):
        max_sp_id = np.max(mask)
        obj_id = obj_ids[m]
        image = get_smaller_image(vid, i)

if __name__ == '__main__':
    run()