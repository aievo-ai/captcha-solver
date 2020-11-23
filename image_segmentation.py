import cv2
import numpy as np


def pre_process(img):
    # Otsu's thresholding after Median filtering
    img = cv2.medianBlur(img, 9)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erode, dilate
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    # More median filtering, erode and dilate to fine tuning
    img = cv2.medianBlur(img, 5)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.medianBlur(img, 5)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    img = cv2.medianBlur(img, 5)
    img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=2)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    return img


def extract_chars(contours):
    char_regions = []

    # Loop through each contour and extract the character
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, width, height) = cv2.boundingRect(contour)
        # Got noise as contour, throw the entire segmentation away
        if width < 10 or height < 10:
            return []
        char_regions.append((x, y, width, height))

    return char_regions


def segment_image(img, num_chars):
    # Convert image to grayscale, blur and then threshold it (convert it to pure black and white)
    pre_processed = pre_process(img)

    # Find the contours in the image
    contours, _ = cv2.findContours(pre_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    char_regions = extract_chars(contours)

    # If there is more regions than chars, the extraction did not work well.
    # Refrain from saving bad data.
    if num_chars > 0:
        if len(char_regions) != num_chars:
            # Could not segment image, return empty array
            return []

    # Sort regions to guarantee that the image represent a specific number
    char_images = sorted(char_regions, key=lambda x: x[0])

    return char_images
