import cv2
from skimage.metrics import peak_signal_noise_ratio
def calculate_psnr(original_image_path, compressed_image_path):
    # Orjinal ve sıkıştırılmış görüntüleri yükle
    original_image = cv2.imread(original_image_path)
    compressed_image = cv2.imread(compressed_image_path)
    # PSNR değerini hesapla
    psnr_value = peak_signal_noise_ratio(original_image, compressed_image)

    return psnr_value

if __name__ == "__main__":
    base_image_path = "a1.png"
    image_to_embed_path = "stego_image.png"
    # PSNR değerini hesapla
    psnr = calculate_psnr(base_image_path, image_to_embed_path)
    print("PSNR Değeri:", psnr)