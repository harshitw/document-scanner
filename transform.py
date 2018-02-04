# four_point_transform
# transform.py
# we can use perspective transform to obtain top down or bird's eye view of an image

import numpy as np
import cv2

def order_points(pts):
    # a list of coordinates such that the first entry in the list is the top-left, the second entry is the top-right, the third is the
    # bottom-right and the fourth is the bottom left
    rect = np.zeros((4, 2), dtype = "float32")
    # top left point will have smallest sum, whereas the bottom right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # computing the difference between the points, top right will have the largest difference while the bottom left will have the smallest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the corrected coordinates
    return rect

def four_point_transform(image, pts):
    # obtaining the consistent order of the points and unpacking them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have dimensions of the new image, we construct a set of destination points to obtain a bird eyes view
    # specifying points in the same order
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

    # computing the perspective transform matrix and then applying it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # returning the wrapped image
    return warped
