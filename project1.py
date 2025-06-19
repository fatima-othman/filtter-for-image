import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def add_watermark(image, text):
    h, w = image.shape[:2]
    x = random.randint(10, w-200)
    y = random.randint(10, h-50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def add_salt_pepper_noise(image, prob=0.05):
    output = image.copy()
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if random.random() < prob:
                output[i, j] = 0 if random.random() < 0.5 else 255
    return output

def main():
    image = cv2.imread('input_image.jpeg')
    if image is None:
        print("Error: Image not found! Please check if 'input_image.jpeg' is in the folder.")
        return

    watermark_text = "Jihad Jamoos - 12216916, Fatima Othman - 12216906"
    watermarked_image = add_watermark(image.copy(), watermark_text)
    cv2.imwrite('watermarked_image.jpeg', watermarked_image)

    gray_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpeg', gray_image)

    h, w = gray_image.shape
    print(f"Dimensions: {w}x{h}")
    print(f"Color channels: 1 (Gray)")
    print(f"Average pixel: {np.mean(gray_image):.2f}")
    print(f"Minimum value: {np.min(gray_image)}")
    print(f"Maximum value: {np.max(gray_image)}")

    # تثبيت قيمة c عشان الهستوغرام ما يتغيرش
    c = 1.5  # اخترت قيمة ثابتة، ممكن تغيرها بين 0.4 و2.0
    bright_image = np.clip(gray_image * c, 0, 255).astype(np.uint8)
    cv2.imwrite('bright_image.jpeg', bright_image)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.hist(gray_image.ravel(), bins=256, range=(0, 255))
    plt.title('Original Histogram')
    plt.subplot(132)
    plt.hist(bright_image.ravel(), bins=256, range=(0, 255))
    plt.title(f'Histogram after Brightness (c={c:.2f})')

    corrected_image = cv2.equalizeHist(bright_image)
    cv2.imwrite('corrected_image.jpeg', corrected_image)

    plt.subplot(133)
    plt.hist(corrected_image.ravel(), bins=256, range=(0, 255))
    plt.title('Histogram after Correction')
    plt.tight_layout()
    plt.savefig('histograms.jpeg')
    plt.show()

    noisy_image = add_salt_pepper_noise(corrected_image)
    cv2.imwrite('noisy_image.jpeg', noisy_image)

    mean_filtered = cv2.blur(noisy_image, (5, 5))
    median_filtered = cv2.medianBlur(noisy_image, 5)
    cv2.imwrite('mean_filtered.jpeg', mean_filtered)
    cv2.imwrite('median_filtered.jpeg', median_filtered)

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(median_filtered, -1, sharpening_kernel)
    cv2.imwrite('sharpened_image.jpeg', sharpened_image)

    print("All photos saved!")

if __name__ == "__main__":
    main()