import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_halftone_effect(img, cell_size=4, dot_intensity=0.368):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gh = max(1, h // cell_size)
    gw = max(1, w // cell_size)
    small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
    cell_mean = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    darkness = 1.0 - (cell_mean / 255.0)
    target_radius = (cell_size / 2.0) * darkness * 0.95

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    cy = (np.floor(y_idx / cell_size) + 0.5) * cell_size
    cx = (np.floor(x_idx / cell_size) + 0.5) * cell_size
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    dot_mask = np.where((dist <= target_radius) & (target_radius >= 0.7), dot_intensity, 1.0).astype(np.float32)

    result = img.astype(np.float32) * dot_mask[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def add_hatching_effect(img, spacing=10, angle=65, shadow_thresh=59, line_intensity=0.397):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    theta = np.deg2rad(angle)
    rotated = x_idx * np.cos(theta) + y_idx * np.sin(theta)
    line_pattern = (np.mod(rotated, spacing) < 1.5)

    shadow_mask = gray < shadow_thresh
    hatching_mask = line_pattern & shadow_mask

    result = img.copy()
    result[hatching_mask] = (result[hatching_mask] * line_intensity).astype(np.uint8)
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
    l_channel, a_channel, b_channel = cv2.split(imgLAB)
    clahe = cv2.createCLAHE(clipLimit=1.21, tileGridSize=(6, 6))
    l_enhanced = clahe.apply(l_channel)
    imgContrast = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    imgQuantized = ((imgContrast.astype(np.int16) // 32) * 32 + 16).astype(np.uint8)

    # N2 saturation
    imgHSV = cv2.cvtColor(imgQuantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    imgHSV[:, :, 1] = np.clip(imgHSV[:, :, 1] * 0.844, 0, 255)
    # imgHSV[:, :, 1] = np.clip(imgHSV[:, :, 1] * 0.78, 0, 255)
    imgSaturated = cv2.cvtColor(imgHSV.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # N3 halftone
    imgHalftone = add_halftone_effect(imgSaturated, cell_size=4, dot_intensity=0.368)

    imgOutlined = imgHalftone.copy()
    imgOutlined[imgEdgesThick > 0] = (0, 0, 0)

    # N4 hatching
    imgFinal = add_hatching_effect(imgOutlined, spacing=10, angle=65, shadow_thresh=59, line_intensity=0.397)

    return imgFinal


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