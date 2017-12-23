import superpixels as sp
import cv2
import imageio
import numpy as np
import pdb
from sklearn.mixture import GaussianMixture
from helpers import *
from graph import Graph

def run():
    filename = './data/toothpick.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Number of frames to use. NONE for all frames.
    num_frames = 2

    # List of superpixel maps
    maps = []

    # List of Edge Adjacency Matrix
    W = []

    # List of ID of superpixel that overlaps with the fixation point
    obj_ids = []

    # Get superpixels and fixation points
    # ================
    for i in get_frames_range(vid, num_frames=num_frames):
        image = get_smaller_image(vid, i)
        sp.num_superpixels = 100
        sp_map = sp.extract_superpixels(image)
        maps.append(sp_map)

        # fixation point is assumed to be at the center of the image.
        fp = sp_map[int(image.shape[0]/2.0), int(image.shape[1]/2.0)]
        obj_ids.append(fp)

    # END - Get superpixels and fixation points
    # ================
    
    # Learn object and background color models
    # ================
    clf_o = GaussianMixture(n_components=1)
    clf_b = GaussianMixture(n_components=1)

    for i in get_frames_range(vid, num_frames=num_frames):
        sp_map = maps[i]
        obj_id = obj_ids[i]

        # TODO: Another alternative is to keep the images in a list. Which would
        #       be better?
        image = get_smaller_image(vid, i)
        
        # fitting the models
        # ------------------
        # Currently, we fit the models by simply including all objects into
        # `clf_o.fit()` and backgrounds to `clf_b.fit()`.

        objects, backgrounds = sp.split_superpixels(image, sp_map, [obj_id])

        for o in objects:
            clf_o.fit(o)
        for b in backgrounds:
            clf_b.fit(b)

        # END - fitting the models
        # ------------------

    # END - Learn object and background color models
    # ================

    # Create W
    # ================
    spm = sp.SuperpixelsMaps(maps)
    W = sp.w_from_superpixels(spm)

    np.savetxt("./data/W.csv", W, delimiter=',')

    # Set normalized weights to all neighboring edges
    # Denumerator is divided by two since the edges are undirectional.
    denumerator = float(W[np.where(W == 1.0)].shape[0]) / 2.0
    W[np.where(W == 1.0)] = 1.0/denumerator
    
    # END - Create W
    # ================

    # generator = visual_hull_not_converged()
    # for convergence in generator:
    #     for img in images:
    #         score = calculate_score(clf_o, clf_b, superpixel)
    #         scores.append(score)

    #     graph_cut(superpixels)
    #     # TODO: Enforce silhouette consistency

    #     # Fitting the models from new silhouettes


if __name__ == '__main__':
    run()