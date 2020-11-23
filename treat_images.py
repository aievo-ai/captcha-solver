import cv2


def load_image(path):
    # Load image, resize and convert color to gray
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR_EXACT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img