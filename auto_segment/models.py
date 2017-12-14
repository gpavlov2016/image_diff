import numpy as np
import math

def process_superpixel(superpixel):
    """ Process superpixel for training data
    """
    return superpixel

def calculate_score(clf_o, clf_b, superpixel):
    score_o = math.exp(clf_o.score(np.array(superpixel)))
    score_b = math.exp(clf_b.score(np.array(superpixel)))
    if (score_o + score_b) == 0:
        return 0
    else:
        return (score_o / (score_o + score_b))

# def train(clf_o, clf_b, image):
#     maps = sp.extract_superpixels(image)
#     sp.