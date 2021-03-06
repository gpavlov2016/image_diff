# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np


#im1 = cv2.normalize(im1, im1, 0, 255, cv2.NORM_MINMAX)
#im2 = cv2.normalize(im2, im2, 0, 255, cv2.NORM_MINMAX)


#im1 = cv2.Canny(im1, 10, 250)
#cv2.imshow('im1', im1)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
import pylab
import imageio
filename = 'toothpick.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
images = []
labels = []
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
for i in range(350): #350
  image = vid.get_data(i)
  fgmask = fgbg.apply(image)
  dilated = cv2.dilate(fgmask, kernel, iterations=10)
  dilated = cv2.erode(dilated, kernel, iterations=10)

  cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]

  mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
  mask = cv2.drawContours(mask, cnts, -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
  out = np.zeros_like(image) # Extract out the object and place into output image
  out[mask == 255] = image[mask == 255]

  cv2.imshow('out', out)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.waitKey(0)
exit(0)


####### Backup code

# Read the images to be aligned
im1 = cv2.imread('im3.jpg')
im2 = cv2.imread('im4.jpg')

im1 = cv2.resize(im1, None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC)
im2 = cv2.resize(im2, None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC)

im1_cpy = im1.copy()
im2_cpy = im2.copy()

im1 = cv2.fastNlMeansDenoisingColored(im1,None,50,50,7,21)
im2 = cv2.fastNlMeansDenoisingColored(im2,None,50,50,7,21)

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Find size of image1
sz = im1.shape

# Define the motion model
warp_mode = cv2.MOTION_HOMOGRAPHY

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY:
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else:
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 1000

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-5

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

#print('cc: ' + cc)
#print('warp_matrix: ' + warp_matrix)

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    # Use warpPerspective for Homography
    im2_aligned_gray = cv2.warpPerspective(im2_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else:
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned_gray = cv2.warpAffine(im2_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

# Show final results
#cv2.imshow("Image 1", im1)
#cv2.imshow("Image 2", im2)
#cv2.imshow("Aligned Image 2", im2_aligned_gray)
#cv2.waitKey(0)

#im2_aligned_gray = im2_gray


grayA = im1_gray
grayB = im2_aligned_gray

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, multichannel=True, gaussian_weights=True, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))



#denoised = cv2.fastNlMeansDenoisingColored(diff,None,50,10,7,21)
denoised = cv2.fastNlMeansDenoising(diff,None,50,7,21)


diff = denoised #cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)



# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
#thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(diff,(3,3),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)
#th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)

thresh = th3
#cv2.imshow("denoised", denoised)
#cv2.waitKey(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
dilated = cv2.dilate(thresh, kernel)

cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for i, c in enumerate(cnts):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ

    (x, y, w, h) = cv2.boundingRect(c)
    #cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

mask = np.zeros_like(im1_cpy) # Create mask where white is what we want, black otherwise
mask = cv2.drawContours(mask, cnts, -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
out = np.zeros_like(im1_cpy) # Extract out the object and place into output image
out[mask == 255] = im1_cpy[mask == 255]

#cv2.imshow('mask', mask)
cv2.imshow('out', out)

#out = cv2.Canny(out, 10, 250)
#cv2.imshow('out', out)


# show the output images
#cv2.imshow("Original", im1)
#cv2.imshow("Modified", im2)
#cv2.imshow("Diff", diff)
#cv2.imshow("Contours", out)
#cv2.imshow("Thresh", thresh)
#cv2.imshow("Diff", diff)
#cv2.imshow("th3", th3)
cv2.waitKey(0)