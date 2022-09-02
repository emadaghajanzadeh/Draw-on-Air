import cv2 as cv
import numpy as np
# cv.namedWindow('image', cv.WINDOW_NORMAL)
cap = cv.VideoCapture(0)
former_cx = 0
former_cy = 0
cx = 0
cy = 0
_,old_frame=cap.read()
mask1 = np.zeros_like(old_frame)
count = 0 ;
while(True):
    ret, image = cap.read()
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_blue = np.array([40, 40, 150])
    upper_blue = np.array([90, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask,kernel,iterations = 1)
    mask = cv.dilate(mask,kernel,iterations = 2)

    res = cv.bitwise_and(image, image, mask=mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if contours!=():
        best_cnt = max(contours, key=cv.contourArea)

        if best_cnt!=[]:
            count = count + 1

            M = cv.moments(best_cnt)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            dist =  (cx-former_cx)**2 + (cy-former_cy)**2

            if(cv.contourArea(best_cnt)>50 and count!=1 and dist < 100 ):
                     mask1 = cv.line(mask1, (cx, cy), (former_cx, former_cy), (0, 255, 0), thickness=2)
            if(cv.contourArea(best_cnt)>4000):
                    mask1 = np.zeros_like(old_frame)

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


