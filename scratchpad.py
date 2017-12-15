import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r = img1.shape[0]
    c = img1.shape[1]
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2

def match_features(img1, img2, intrinsics, dist):
    w = img1.shape[1]
    h = img1.shape[0]
    # Initiate ORB detector
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=1000)
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    #print('kp1: ', kp1[0].pt)
    #print('des1:', des1)
    kp2, des2 = orb.compute(img2, kp2)
    # draw only keypoints location,not size and orientation
    vis = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    #plt.imshow(vis), plt.show()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = np.zeros_like(img2)
    # Draw first 10 matches.
    n_matches = 100
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_matches], img3, flags=2)
    #plt.imshow(img3), plt.show()

    #findBaselineTriangulation
    pleft = np.eye(3,4)
    pright = np.eye(3,4)

    focal = intrinsics[0,0]
    ppx = intrinsics[0,2]
    ppy = intrinsics[1,2]
    alignedLeft = {'pt': [], 'des': []}
    alignedRight = {'pt': [], 'des': []}
    leftBackReference = []
    rightBackReference = []
    #Arrange matching points in aligned arrays:
    for i, match in enumerate(matches[:n_matches]):
        qid = match.queryIdx
        tid = match.trainIdx
        alignedLeft['pt'].append(kp1[qid].pt)
        alignedLeft['des'].append(des1[qid,:])
        alignedRight['pt'].append(kp2[tid].pt)
        alignedRight['des'].append(des2[tid,:])
    pts1 = np.array(alignedLeft['pt']).reshape(n_matches, 1, -1).astype(np.float32)
    pts2 = np.array(alignedRight['pt']).reshape(n_matches, 1, -1).astype(np.float32)


    #plt.subplot(221), plt.scatter(pts1[:,0,0], pts1[:,0,1]), plt.title('pts1'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    #plt.subplot(222), plt.scatter(pts2[:,0,0], pts2[:,0,1]), plt.title('pts2'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    #plt.subplot(223), plt.imshow(img1, 'gray'), plt.title('img1')
    #plt.subplot(224), plt.imshow(img2, 'gray'), plt.title('img2')
    #plt.show()

    '''
    pts_uv1 = cv2.undistortPoints(pts1, intrinsics, distCoeffs=None, P=intrinsics)
    pts_uv2 = cv2.undistortPoints(pts2, intrinsics, distCoeffs=None, P=intrinsics)

    for pt in pts_uv1[:,0,:]:
        cv2.circle(img1, (pt[0], pt[1]), 10, (255), -1)
    for pt in pts_uv2[:,0,:]:
        cv2.circle(img2, (pt[0], pt[1]), 10, (255), -1)


    plt.subplot(221), plt.scatter(pts_uv1[:,0,0], pts_uv1[:,0,1]), plt.title('pts1'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    plt.subplot(222), plt.scatter(pts_uv2[:,0,0], pts_uv2[:,0,1]), plt.title('pts2'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    plt.subplot(223), plt.imshow(img1, 'gray'), plt.title('img1')
    plt.subplot(224), plt.imshow(img2, 'gray'), plt.title('img2')
    #plt.show()

    pts_uv1_t = np.reshape(pts_uv1, (1,-1,2))[0,0:4,:]
    pts_uv2_t = np.reshape(pts_uv2, (1,-1,2))[0,0:4,:]
    print('pts_uv1_t:', pts_uv1_t, pts_uv1_t.dtype)
    print('pts_uv2_t:', pts_uv2_t, pts_uv2_t.dtype)

    pts_uv1_t = np.reshape(pts1, (1,-1,2))[0,0:4,:]
    pts_uv2_t = np.reshape(pts2, (1,-1,2))[0,0:4,:]

    M = cv2.getPerspectiveTransform(pts_uv1_t, pts_uv2_t)

    print('M: ', M)

    imt = cv2.imread('chess.jpg')

    #M = np.eye(3)
    dst = cv2.warpPerspective(imt, M, (imt.shape[1], imt.shape[0]))
    #print(dst)

    plt.subplot(121), plt.imshow(img1, 'gray'), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()



    retval, mask = cv2.findHomography(pts1, pts2)
    print(retval, mask)
    M = cv2.estimateRigidTransform(pts1, pts2, True)
    print(M)

    dst = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    #print(dst)

    plt.subplot(131), plt.imshow(img1), plt.title('img1')
    plt.subplot(132), plt.imshow(img2), plt.title('img2')
    plt.subplot(133), plt.imshow(dst), plt.title('img1 warped')
    plt.show()

    exit(0)

    '''

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    print('F: ', F)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    exit(0)

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=(ppx, ppy))
    print('E: ', E)
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=(ppx, ppy), mask=mask)
    print('points: ', points)
    print('R: ', R)
    print('t: ', t)
    print('mask: ', mask)
    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))


    print(M_r, M_r.shape)

    print('angles:',rotationMatrixToEulerAngles(R))

    #triangulate:

    P_l = np.dot(intrinsics, M_l)
    P_r = np.dot(intrinsics, M_r)
    point_4d_hom = cv2.triangulatePoints(M_l, M_r, pts1, pts2)
    print('point_4d_hom: ', point_4d_hom)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T

    print(point_3d)

    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    import random


    fig = pyplot.figure()
    ax = Axes3D(fig)

    sequence_containing_x_vals = point_3d[:,0]
    sequence_containing_y_vals = point_3d[:,1]
    sequence_containing_z_vals = point_3d[:,2]

    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    pyplot.show()





def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows = 7
    cols = 7
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    import pylab
    import imageio
    #filename = 'calib/checkerboard.mp4'
    #vid = imageio.get_reader(filename,  'ffmpeg')
    images = []
    #for i in range(10):
    #  image = vid.get_data(i)
    #  images.append(image)
    #  #cv2.imwrite('calib/frame_' + str(i) + '.jpg', image)
    import glob

    #for i in range(0, len(images), 1):
    #images = ['IMG_20171211_151001.jpg']
    images = glob.glob('calib/*.jpg')
    print(images)
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img', gray)
        #cv2.waitKey(500)

        print('Processing calibration images')
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    if len(images) == 0:
        print("Put images into calib directory.")
    return  cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def from_video():
    import pylab
    import imageio
    filename = 'toothpick.mp4'
    vid = imageio.get_reader(filename,  'ffmpeg')
    images = []
    for i in range(210): #350
      image = vid.get_data(i)
      images.append(image)

    imgL = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)#cv2.imread('tsukuba_l.png',0)
    imgR = cv2.cvtColor(images[150], cv2.COLOR_BGR2GRAY)#cv2.imread('tsukuba_r.png',0)

    plt.figure()
    plt.imshow(imgL, 'gray')
    plt.figure()
    plt.imshow(imgR, 'gray')
    plt.show()


ret, mtx, dist, rvecs, tvecs = calibrate()
print('Intrinsic Matrix: ', mtx)


imgL = cv2.imread('im1.jpg')
imgR = cv2.imread('im2.jpg')

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

match_features(imgL, imgR, mtx, dist)





exit(0)




img = cv2.imread('calib/IMG_20171211_151001.jpg')
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
plt.figure()
plt.imshow(dst, 'gray')
plt.show()

imgL = cv2.cvtColor(cv2.imread('im1.png'), cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(cv2.imread('im2.png'), cv2.COLOR_BGR2GRAY)

imgL = cv2.remap(imgL,mapx,mapy,cv2.INTER_LINEAR)
imgR = cv2.remap(imgR,mapx,mapy,cv2.INTER_LINEAR)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.figure()
plt.imshow(disparity,'gray')
print(disparity)
plt.show()