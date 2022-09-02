import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
#The following variables (cx and cy) are being used for drawing lines
former_cx = 0
former_cy = 0
cx = 0
cy = 0
#Capture a frame for initial settings
_,old_frame=cap.read()
mask1 = np.zeros_like(old_frame)
FirstCaputure = True
#Please select the proper range here
lowerBound = np.array([10,40,150])
upperBound = np.array([90,255,255])

#The Loop which resposible for capturing and processing the frames
while(True):
    #Capturing and initial processing
    ret, image = cap.read()
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lowerBound, upperBound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask,kernel,iterations = 1)
    mask = cv.dilate(mask,kernel,iterations = 2)

    res = cv.bitwise_and(image, image, mask=mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if contours!=():
        best_cnt = max(contours, key=cv.contourArea)

        #If there is a best contour then draw a line associated with the current position
        if best_cnt!=[]:
            M = cv.moments(best_cnt)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            dist =  (cx-former_cx)**2 + (cy-former_cy)**2

            if(cv.contourArea(best_cnt)>50 and (not FirstCaputure) and (dist < 100) ):
                     mask1 = cv.line(mask1, (cx, cy), (former_cx, former_cy), (0, 255, 0), thickness=2)
            if(cv.contourArea(best_cnt)>4000):
                    mask1 = np.zeros_like(old_frame)

            FirstCaputure = False

    img = cv.add(image, mask1)
    img_flip_ud = cv.flip(img, 1)
    cv.imshow('frame', img_flip_ud)
    cv.imshow('mask', mask)
    former_cx = cx
    former_cy = cy

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


