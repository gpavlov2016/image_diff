import cv2
print(cv2.__version__)
cap = cv2.VideoCapture('adapter.mp4')

if (cap.isOpened()==False):
    print("Error opening video file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame',frame)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()