import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_halftone_effect(img, cell_size=4, dot_intensity=0.368):
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = img.copy()

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):

            cell = gray[y:y+cell_size, x:x+cell_size]

            mean = np.mean(cell)
            darkness = 1.0 - mean / 255.0

            radius = (cell_size / 2.0) * darkness * 0.95

            if radius >= 0.7:
                cx = x + cell_size // 2
                cy = y + cell_size // 2

                for dy in range(-cell_size // 2, cell_size // 2 + 1):
                    for dx in range(-cell_size // 2, cell_size // 2 + 1):

                        px = cx + dx
                        py = cy + dy

                        if 0 <= px < w and 0 <= py < h:
                            distance = np.sqrt(dx**2 + dy**2)

                            if distance <= radius:
                                result[py, px] = result[py, px] * dot_intensity

    return result

def comic_filter(img):
    imgGaussian = cv2.GaussianBlur(img, (3, 3), 0)

    imgGray = cv2.cvtColor(imgGaussian, cv2.COLOR_BGR2GRAY)
    imgEdges = cv2.Canny(imgGray, 39, 142)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    imgEdgesThick = cv2.dilate(imgEdges, kernel, iterations=1)

    imgMedian = cv2.medianBlur(imgGaussian, 5)

    # N1 contrast
    imgLAB = cv2.cvtColor(imgMedian, cv2.COLOR_BGR2LAB)
    l_channel = imgLAB[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=1.21, tileGridSize=(6, 6))
    l_enhanced = clahe.apply(l_channel)
    imgLAB[:, :, 0] = l_enhanced
    imgContrast = cv2.cvtColor(imgLAB, cv2.COLOR_LAB2BGR)

    imgQuantized = ((imgContrast.astype(np.int16) // 32) * 32 + 16).astype(np.uint8)

    # N2 saturation
    imgHSV = cv2.cvtColor(imgQuantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_channel = imgHSV[:, :, 1]
    s_channel = s_channel * 0.844
    s_channel = np.clip(s_channel, 0, 255)
    imgHSV[:, :, 1] = s_channel
    imgSaturated = cv2.cvtColor(imgHSV.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # N3 halftone
    imgHalftone = add_halftone_effect(imgSaturated, cell_size=4, dot_intensity=0.368)

    imgOutlined = imgHalftone.copy()
    imgOutlined[imgEdgesThick > 0] = (0, 0, 0)

    return imgOutlined


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