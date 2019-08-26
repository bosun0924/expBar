import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import the image
img = cv2.imread('./1.png')
################### Edge Detection ###################
# Gray the Image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the Image
blur = cv2.blur(gray, (5,5))
# Canny Edge Detection
edge = cv2.Canny(blur, 50, 130)

################### Find the head circle ###################
circles = cv2.HoughCircles(edge, #input picture
                            cv2.HOUGH_GRADIENT,
                            1,500,
                            param1=50,param2=30,
                            minRadius=48,maxRadius=62)
print(circles)

for i in circles[0,:]:
    if (588 <= i[0] <= 596)and(996 <= i[0] <= 1000): 
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

################ Region of Interest(ROI) ################
circle = circles[0][0]#extract the circle
print (circle)
ROI = img[int(circle[1]-circle[2]-5):int(circle[1]+circle[2]+5),int(circle[0]+20):int(circle[0]+circle[2]+20)]

# Show the result
result = ROI

plt.imshow(result)
plt.show()
