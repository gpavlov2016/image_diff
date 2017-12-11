import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
import imutils
import pdb

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

trans_counter = 1
trans_output_dir = os.path.join('data', 'edges', str(time.time()))
if os.path.exists(trans_output_dir):
    shutil.rmtree(trans_output_dir)
os.makedirs(trans_output_dir)

def save_transformed_image(image):
    global trans_counter, trans_output_dir
    name = "{}.png".format(trans_counter)
    scipy.misc.imsave(os.path.join(trans_output_dir, name), image)
    trans_counter += 1

def apply_transformation(image, image_shape):
    image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY );
    image = cv2.Canny(image, 200, 100)
    save_transformed_image(image)

    boolmask = (image == 255)
    label = boolmask.reshape(*boolmask.shape, 1)
    label = np.concatenate((label, np.invert(label)), axis=2)
    return label

def process_image(image, images, labels, image_shape):
    import numpy as np
    #image[(image[:,:,0] >100) & (image[:,:,1] > 100) & (image[:,:,2] > 100)] = 0
    #pylab.imshow(image)
    #pylab.show()
    image_r = scipy.misc.imresize(image, image_shape)
    #pylab.imshow(image_r)
    #label = np.zeros_like(image_r)
    #background_color = np.array([255, 0, 0])
    #label[:,:] = background_color
    # threshold = ((image_r[:,:,0] < 100) | (image_r[:,:,1] < 100) | (image_r[:,:,2] < 100))
    label = apply_transformation(image_r, image_shape)

    images.append(image_r)
    labels.append(label)

# def prepare_training_data(vid, image_shape=None):
#     images = []
#     labels = []
#     fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
#     for i in range(10): #350
#         image = vid.get_data(i)
#         if image_shape is not None:
#             image = scipy.misc.imresize(image, image_shape)
#         fgmask = fgbg.apply(image)
#         dilated = cv2.dilate(fgmask, kernel, iterations=5)
#         dilated = cv2.erode(dilated, kernel, iterations=5)

#         cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#         mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
#         mask = cv2.drawContours(mask, cnts, -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
#         out = np.zeros_like(image) # Extract out the object and place into output image
#         out[mask == 255] = image[mask == 255]

#         label = np.array([[[True, False]]*image.shape[1]]*image.shape[0])
#         boolmask = (mask == [255, 255, 255]).all(axis=-1)
#         label[boolmask] = [False, True]
#         images.append(image)
#         labels.append(label)
#     return (images, labels)

def gen_batch_function(vid, image_shape):
    """
    Generate function to create batches of training data
    :param vid: Video object gathered by `imageio.get_reader()`
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        
        images = []
        labels = []

        for i in range(200):
            image = vid.get_data(i)
            process_image(image, images, labels, image_shape)

        #random.shuffle(image_paths)
        for batch_i in range(0, len(images), batch_size):
            #print('batch ' + str(batch_i))
            yield images[batch_i:batch_i+batch_size], labels[batch_i:batch_i+batch_size]
    return get_batches_fn

def get_validation_set(vid, image_shape):
    import pylab
    import imageio
    images = []
    labels = []
    for i in range(10):
        image = vid.get_data(i)
        process_image(image, images, labels, image_shape)
    return images, labels

def get_inference_set(vid, image_shape):
    """ Get images from video for inference.
    """
    import pylab
    import imageio
    images = []
    labels = []
    for i in range(10):
        image = vid.get_data(i)
        process_image(image, images, labels, image_shape)
    return images


def gen_test_output(vid, sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param vid: Video object
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    
    images = get_inference_set(vid, image_shape)
    #for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
    for i in range(len(images)):
        #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        image = images[i]
        image_file = 'frame_' + str(i) + '.png'
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(vid, runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        vid, sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        print(name)
        scipy.misc.imsave(os.path.join(output_dir, name), image)
