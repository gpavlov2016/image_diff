import cv2

def get_smaller_image(vid, i):
    image = vid.get_data(i)
    image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))
    return image

def show_image(image):
    cv2.imshow('out', image)

    cv2.waitKey(0)