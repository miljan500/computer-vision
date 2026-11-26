import cv2
import numpy as np
import matplotlib.pyplot as plt

def comic_filter(img):
    imgGaussian = cv2.GaussianBlur(img, (5, 5), 0)

    imgGray = cv2.cvtColor(imgGaussian, cv2.COLOR_BGR2GRAY)
    imgEdges = cv2.Canny(imgGray, 38, 141)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    imgEdges = cv2.dilate(imgEdges, kernel, iterations=1)

    imgMedian = cv2.medianBlur(img, 5)

    imgHSV = cv2.cvtColor(imgMedian, cv2.COLOR_BGR2HSV).astype(np.float32)
    imgHSV[:, :, 1] = np.clip(imgHSV[:, :, 1] * 1.35, 0, 255)
    imgVibrant = cv2.cvtColor(imgHSV.astype(np.uint8), cv2.COLOR_HSV2BGR)

    imgQuantized = ((imgVibrant.astype(np.int16) // 32) * 32 + 16).astype(np.uint8)

    imgQuantized[imgEdges > 0] = (0, 0, 0)

    return imgQuantized


if __name__ == '__main__':
    imgIn = cv2.imread("input-image.png")

    if imgIn is None:
        print("Image not found!")
        exit()

    imgOut = comic_filter(imgIn)
    cv2.imwrite("output-image.png", imgOut)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB))
    plt.title("Originalna slika")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imgOut, cv2.COLOR_BGR2RGB))
    plt.title("Comic filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()