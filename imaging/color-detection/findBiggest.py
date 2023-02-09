import cv2 as cv
import argparse
import imutils
# Use findBiggest.py -i PATH_TO_IMAGE
# construct argument parse and parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the input image")
args = vars(ap.parse_args())

#load image and convert it to grayscale, blur it slightly
img = cv.imread(args["image"])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (25,25,), 0)
thresh = cv.threshold(blur, 60, 255, cv.THRESH_BINARY)[1]

cv.imshow("blurred", blur)
# find contours in the thresholded image

contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # Returns set of otulines corresponding to each shape
contours = imutils.grab_contours(contours) #grabs appropriate contour?

# Loop thru contours
biggest = 0
biggestShape = None
for c in contours:
    #find center of contour
    area = cv.contourArea(c)
    if(area > biggest):
        biggest = area
        biggestShape = c
       

c = biggestShape
M = cv.moments(c)
print(M["m10"] , M["m00"])
if(M["m00"] != 0 ):
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    #Draw contour and center of shape onto image
    cv.drawContours(img, [biggestShape], -1, (0,255,0), 2)
    cv.circle(img, (center_x, center_y), 7, (255, 255, 255), -1)
    cv.putText(img, "center", (center_x - 20, center_y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)






cv.imshow("thresh",img)
while True:
    if cv.waitKey(100) & 0xFF==ord('d'): # & is used for bits
        break

cv.destroyAllWindows()