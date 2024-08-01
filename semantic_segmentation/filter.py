import numpy as np
import cv2


def keep_biggest_blob(semSegInfPath) -> np.ndarray:
    img = cv2.imread(semSegInfPath, cv2.IMREAD_GRAYSCALE)
    # ? img = np.uint8(img > 0)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestAreaContour = max(contours, key=cv2.contourArea)
    biggestBlobMask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(biggestBlobMask, [largestAreaContour], 0, 255, cv2.FILLED)  # type: ignore
    img = cv2.bitwise_and(img, biggestBlobMask)
    return img
