import cv2

def get_smaller_image(vid, i, ratio=0.25):
    image = vid.get_data(i)
    image = cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
    return image

def show_image(image):
    cv2.imshow('out', image)

    cv2.waitKey(0)

def get_frames_range(vid, num_frames=None):
    """ Get list of frame indexes from a video.
    """
    return range(num_frames)


def visual_hull_not_converged():
    """ TODO: Not sure what this is for, so we will just use range here.
              Stops after iterates 2 times.
    """
    value = True
    for i in range(2):
        if i == 2:
            value = False
        else:
            value = True
        yield(value)
