import cv2
import numpy as np


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def biFilter(image):
    params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

    # loop over the diameter, sigma color, and sigma space
    for (diameter, sigmaColor, sigmaSpace) in params:
        # apply bilateral filtering to the image using the current set of
        # parameters
        blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)

        # show the output image and associated parameters
        title = "Blurred d={}, sc={}, ss={}".format(
            diameter, sigmaColor, sigmaSpace)
        return blurred
        # cv2.imshow(title, blurred)
        # cv2.waitKey(0)


def sharpen(image, kernelIndex=6):
    kernels = [
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        (1 / 9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        (1 / 256) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                             [1, 4, 6, 4, 1]]),
        (-1 / 256) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4],
                              [1, 4, 6, 4, 1]])
    ]

    # for index, kernel in enumerate(kernels):
    #     sharpen_kernel = kernel
    #     sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    #     cv2.imshow(str(index), sharpen)
    #     cv2.waitKey(0)
    sharpen = cv2.filter2D(image, -1, kernels[kernelIndex])
    return sharpen
