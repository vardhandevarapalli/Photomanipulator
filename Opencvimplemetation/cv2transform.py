import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    adjusted_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return adjusted_image

def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def apply_vignette_effect(image: np.ndarray, alpha: float) -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_coords = np.dstack(np.mgrid[0:height, 0:width]) / max_dist
    vignette_image = image * (1 - alpha * normalized_coords**2)
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)
    return vignette_image

def blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
    return blurred_image

def blur2(image: np.ndarray, kernel_size: int) -> np.ndarray:
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image

def brighten(image: np.ndarray, factor: float) -> np.ndarray:
    brightened_image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return brightened_image

def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    ssim = cv2.compareSSIM(image1, image2)
    return ssim

def cluster_pixels(image: np.ndarray, num_clusters: int) -> np.ndarray:
    reshaped_image = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(reshaped_image, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return segmented_image

def color_dodge(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    blended_image = cv2.divide(image1, 255 - image2, scale=256.0)
    return blended_image

def colorize(image: np.ndarray, color_palette: list) -> np.ndarray:
    colorized_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return colorized_image

def compress_image(image: np.ndarray, compression_factor: int) -> np.ndarray:
    _, compressed_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), compression_factor])
    decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
    return decompressed_image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def crop(image: np.ndarray, x_start: int, y_start: int, width: int, height: int) -> np.ndarray:
    cropped_image = image[y_start:y_start+height, x_start:x_start+width]
    return cropped_image

def edge_detection_x(image: np.ndarray) -> tuple:
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return sobel_x

def edge_detection_y(image: np.ndarray) -> tuple:
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_y

def image_blending(image1: np.ndarray, image2: np.ndarray, alpha: float) -> np.ndarray:
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended_image

def image_comparison(image1: np.ndarray, image2: np.ndarray) -> tuple:
    mse = np.mean((image1 - image2) ** 2)
    ssim = calculate_ssim(image1, image2)
    return mse, ssim

def image_compression(image: np.ndarray, compression_factor: int) -> np.ndarray:
    compressed_image = compress_image(image, compression_factor)
    return compressed_image

def image_drawing(image: np.ndarray, x: list, y: list, color: tuple, thickness: int, alpha: float) -> np.ndarray:
    drawn_image = image.copy()
    for i in range(len(x)):
        cv2.line(drawn_image, (x[i], y[i]), (x[i+1], y[i+1]), color, thickness)
    blended_image = image_blending(image, drawn_image, alpha)
    return blended_image

def image_segmentation(image: np.ndarray, num_clusters: int) -> np.ndarray:
    segmented_image = cluster_pixels(image, num_clusters)
    return segmented_image

def image_sharpening(image: np.ndarray, alpha: float) -> np.ndarray:
    blurred_image = apply_gaussian_blur(image, 5)
    sharpened_image = cv2.addWeighted(image, 1 + alpha, blurred_image, -alpha, 0)
    return sharpened_image

def img_as_ubyte(image: np.ndarray) -> np.ndarray:
    scaled_image = np.clip(image, 0, 255).astype(np.uint8)
    return scaled_image

def pencil_sketch(image: np.ndarray, factor: float = 256.0) -> np.ndarray:
    grayscale_image = convert_to_grayscale(image)
    inverted_image = cv2.bitwise_not(grayscale_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    pencil_sketch_image = cv2.divide(grayscale_image, blurred_image, scale=factor)
    return pencil_sketch_image

def plot_histogram(image: np.ndarray, save_path: str) -> None:
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.savefig(save_path)
    plt.close()

def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def resize2(image: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = cv2.INTER_CUBIC
    resized_image = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized_image

def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image
