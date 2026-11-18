
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft


def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    return img_filtered


def my_max_filter(img, size=5):
    rows, cols = img.shape
    pad = size // 2

    padded = np.pad(img, pad, mode='edge')
    result = np.zeros_like(img)

    for y in range(rows):
        for x in range(cols):
            window = padded[y:y + size, x:x + size]
            result[y, x] = np.max(window)

    return result


def ukloni_periodicni_sum(img_gray):
    img_fft = fft(img_gray)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)

    magnitude_log_pre = img_fft_log.copy()

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    center_mask = (x - ccol) ** 2 + (y - crow) ** 2 <= 5 ** 2

    local_max = my_max_filter(img_fft_log, size=5)

    threshold = img_fft_log.mean() + 5 * img_fft_log.std()

    peaks = (img_fft_log == local_max) & (img_fft_log > threshold) & (~center_mask)

    img_fft_log[peaks] = 0

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered, magnitude_log_pre, img_fft_log


if __name__ == '__main__':
    input_folder = "Zadatak 4"
    output_folder = "Zadatak 4_resenja"

    os.makedirs(output_folder, exist_ok=True)

    ekstenzije = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' ne postoji!")
    else:
        fajlovi = sorted(os.listdir(input_folder))

        for ime_fajla in fajlovi:
            if not ime_fajla.lower().endswith(ekstenzije):
                continue

            putanja = os.path.join(input_folder, ime_fajla)
            img = cv2.imread(putanja, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            ociscena_slika, spektar_pre, spektar_posle = ukloni_periodicni_sum(img)

            izlazna_putanja = os.path.join(output_folder, f"filtered_{ime_fajla}")
            cv2.imwrite(izlazna_putanja, ociscena_slika)
            print(f"Sacuvano: {izlazna_putanja}")

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"Rezultat za: {ime_fajla}", fontsize=14)

            axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_title("Originalna slika (sa sumom)")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(spektar_pre, cmap='gray')
            axes[0, 1].set_title("Amplitudni spektar (pre)")
            axes[0, 1].axis('off')

            axes[1, 0].imshow(ociscena_slika, cmap='gray')
            axes[1, 0].set_title("Filtrirana slika")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(spektar_posle, cmap='gray')
            axes[1, 1].set_title("Amplitudni spektar (posle uklanjanja pikova)")
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.show()
