import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img1):
    img_fft = np.fft.fft2(img1)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    return img_filtered

def ukloni_periodicni_sum(img_gray):
    img_fft = fft(img_gray)

    img_fft_mag = np.abs(img_fft)

    img_mag_1 = img_fft / img_fft_mag

    img_fft_log = np.log(img_fft_mag)

    magnitude_log_pre = img_fft_log.copy()

    rows, cols = img_gray.shape
    crow = rows // 2
    ccol = cols // 2

    center = (crow, ccol)
    radius = 5

    center_mask = np.zeros((rows, cols), dtype=bool)

    for x in range(rows):
        for y in range(cols):
            if (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1]) <= radius * radius:
                center_mask[x, y] = True

    threshold = img_fft_log.mean() + 6.14 * img_fft_log.std()

    peaks = (img_fft_log > threshold) & (~center_mask)

    img_fft_log[peaks] = 0

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered, magnitude_log_pre, img_fft_log

if __name__ == '__main__':
    input_folder = "Slike Ulazne"
    output_folder = "Slike Filtrirane"

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' ne postoji!")
    else:
        fajlovi = sorted(os.listdir(input_folder))

        for ime_fajla in fajlovi:

            putanja = os.path.join(input_folder, str(ime_fajla))
            img = cv2.imread(putanja)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ociscena_slika, spektar_pre, spektar_posle = ukloni_periodicni_sum(img)

            izlazna_putanja = os.path.join(output_folder, f"filtered_{ime_fajla}")

            ociscena_slika = np.clip(ociscena_slika, 0, 255).astype(np.uint8)
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
