import cv2

img = cv2.imread('im4.jpg')

img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY );
img = cv2.blur( img, (3,3) );
img = cv2.Canny(img, 200, 100)

cv2.imshow('', img);
cv2.waitKey(0)
exit(0)