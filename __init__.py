import cv2
import numpy as np
import matplotlib.pyplot as plt
from digitize import *

def main():
    ################### 1.Import the image ####################
    img = cv2.imread('./test.png')
    ################### 2.Edge Detection ###################
    # Gray the Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the Image
    blur = cv2.blur(gray, (5,5))
    # Canny Edge Detection
    edge = cv2.Canny(blur, 50, 130)

    ################### 3.Find the head circle ###################
    circles = cv2.HoughCircles(edge, #input picture
                                cv2.HOUGH_GRADIENT,
                                1,500,
                                param1=50,param2=30,
                                minRadius=48,maxRadius=62)
    #print(circles)
    '''
    if circles is not None:
        for i in circles[0,:]:
            if (588 <= i[0] <= 596)and(996 <= i[0] <= 1000): 
                # draw the outer circle
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    '''
    ################ 4.Region of Interest(ROI) ################
    circle = circles[0][0]#extract the circle
    #print (circle)
    ROI = img[int(circle[1]-circle[2]-5):int(circle[1]+circle[2]+5),int(circle[0]+20):int(circle[0]+circle[2]+20)]

    ################ 5.Extract the EXP bar #################
    # Transform to HSV color space
    ROI_HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    # inRange Thresholding(extract color purple) to extract the bar
    dark_purple = (130, 160, 0)
    light_purple = (140, 255, 255)
    exp_extracting = cv2.inRange(ROI_HSV, dark_purple, light_purple)

    ############### 6.Digitaize the Exp Bar ################
    lum, _, _, _ = cv2.sumElems(exp_extracting)
    ## Calibration ##
    # about every 2213.4 of brightness makes 1% of EXP to the next level
    progress = int(lum/2213.4)
    print('Exp Percentage to the next level: {0}%'.format(progress))

    ##############_______________Exp Level_______________################
    ############## 1. Find the Number Circle ##############
    ROI_gray =  cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI_blur = cv2.blur(ROI_gray, (5,5))
    ROI_edge = cv2.Canny(ROI_blur, 20, 30)
    exp_circles = cv2.HoughCircles(ROI_edge, #input picture
                                cv2.HOUGH_GRADIENT,
                                1,50,
                                param1=30,param2=20,
                                minRadius=10,maxRadius=20)
    if exp_circles is not None:
        for i in exp_circles[0,:]: 
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    #print(exp_circles)
    ############## 2. Crop the Number area ################
    low = (0, 0, 200)
    high = (180, 5, 255)
    level_ext = cv2.inRange(ROI_HSV, low, high)
    ############## 3. Textualize the Number ###############
    print(func1(level_ext))
    #################### Images ###################
    result = exp_extracting

    cv2.imshow('level_ext',level_ext)
    cv2.waitKey(0)
    '''
    plt.imshow(level_ext)
    plt.show()
    
    cv2.imshow('result', result)
    cv2.imshow('ROI', ROI_edge)
    cv2.waitKey(0)
    '''

if __name__ == '__main__':
    main()

