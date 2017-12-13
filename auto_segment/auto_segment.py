import superpixels as sp
import cv2
import imageio


def run():
    filename = 'toothpick.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    # List of superpixels
    S = []

    # List of Edge Adjacency Matrix
    W = []

    # List of ID of superpixel that overlaps with the fixation point
    fps = []

    for i in range(1):
        image = vid.get_data(i)

        image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))
        sp.num_superpixels = 100
        superpixels = sp.extract_superpixels(image)
        S.append(superpixels)

        # fixation point is assumed to be at the center of the image.
        fp = (image.shape[0]/2.0, image.shape[1]/2.0)

        mask = np.zeros(img.shape[:2], np.uint8)
        mask[100:300, 100:400] = 255
        masked_img = cv2.bitwise_and(img,img,mask = mask)

    # Learn background color model


if __name__ == '__main__':
    run()