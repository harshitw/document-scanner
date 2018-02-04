# kick ass document scanner

# scanning the document is done in three steps
# detect edges, use edges in the image to find the contours representing the piece of document to be scanned, apply perspective tranform to obtain
# the top down view of of the document
# perspective transform is also used in the image segmentation

# scan.py

# import the necessary packages
from vision_lib.transform import four_point_transform
from vision_lib import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2

# consturct the argument parser and parse the Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# edge detection
image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 80, 180)

# showing the original image and the edge detected image
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Second step is the contour detection
# heuristic to detect contour will be largest contour with four points
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# CHAIN_APPROX_SIMPLE saves memory by saving only few points in contours for eg in case of rectangle it stores four points
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# CHECK THIS LINE [:5] NOT CLEAR

# loop over the contours
for c in cnts:
    # approximate the appropriate contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # we approximated our contour has four points
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# now we will apply a perspective transform and threshold
# applying the four point transform to obtain a birds eye view of the image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# connverting the image into grayscale to give white and black effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8")*255

cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
