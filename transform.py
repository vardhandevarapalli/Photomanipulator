import os
import cv2
import math
import numpy as np
from image import Image
#from imagealt import Image
import matplotlib.pyplot as plt
from PIL import Image as Img
from PIL import ImageDraw as ImgDraw
from PIL import ImageFilter as ImgFilter
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.util import img_as_ubyte
def brighten(image, factor):
    # when we brighten, we just want to make each channel higher by some amount 
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!

    # # this is the non vectorized version
    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             new_im.array[x, y, c] = image.array[x, y, c] * factor

    # faster version that leverages numpy
    new_im.array = image.array * factor

    return new_im

def adjust_contrast(image, factor, mid):
    # adjust the contrast by increasing the difference from the user-defined midpoint by factor amount
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid

    return new_im

def blur(image, kernel_size):
    # kernel size is the number of pixels to take into account when applying the blur
    # (ie kernel_size = 3 would be neighbors to the left/right, top/bottom, and diagonals)
    # kernel size should always be an *odd* number
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    neighbor_range = kernel_size // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a kernel of 3, this value should be 1)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                # we are going to use a naive implementation of iterating through each neighbor and summing
                # there are faster implementations where you can use memoization, but this is the most straightforward for a beginner to understand
                total = 0
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)
    return new_im

def apply_kernel(image, kernel):
    # the kernel should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    neighbor_range = kernel.shape[0] // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a 3x3 kernel, this value should be 1)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_im.array[x, y, c] = total
    return new_im
def rotate(image, angle):
    # Rotate the image by a specified angle (in degrees)
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)
    rad_angle = math.radians(angle)
    cos_theta = math.cos(rad_angle)
    sin_theta = math.sin(rad_angle)
    center_x = x_pixels / 2
    center_y = y_pixels / 2

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_x = int((x - center_x) * cos_theta - (y - center_y) * sin_theta + center_x)
                new_y = int((x - center_x) * sin_theta + (y - center_y) * cos_theta + center_y)

                if new_x >= 0 and new_x < x_pixels and new_y >= 0 and new_y < y_pixels:
                    new_im.array[x, y, c] = image.array[new_x, new_y, c]
    
    return new_im

def crop(image, x_start, y_start, width, height):
    # Crop a specific region of interest from an image
    x_end = x_start + width
    y_end = y_start + height
    cropped_array = image.array[x_start:x_end, y_start:y_end, :]
    new_im = Image(x_pixels=width, y_pixels=height, num_channels=image.num_channels)
    new_im.array = cropped_array
    
    return new_im

def resize2(image, new_width, new_height):
    # Resize an image to the specified dimensions
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=new_width, y_pixels=new_height, num_channels=num_channels)
    x_scale = x_pixels / new_width
    y_scale = y_pixels / new_height

    for x in range(new_width):
        for y in range(new_height):
            for c in range(num_channels):
                old_x = int(x * x_scale)
                old_y = int(y * y_scale)
                new_im.array[x, y, c] = image.array[old_x, old_y, c]
    
    return new_im

def colorize(image, color_palette):
    # Convert a grayscale image to a color image using a color palette
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    for x in range(x_pixels):
        for y in range(y_pixels):
            intensity = image.array[x, y, 0]
            color_index = int(intensity * (len(color_palette) - 1))
            new_im.array[x, y, :] = color_palette[color_index]
    
    return new_im

"""def colorize(image, color_palette):
    # Convert a grayscale image to a color image using a color palette
    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=3)

    for x in range(x_pixels):
        for y in range(y_pixels):
            intensity = image.array[x, y, 0]
            color_index = int(intensity * (len(color_palette) - 1))
            new_im.array[x, y, :] = color_palette[color_index]
    
    return new_im"""

def edge_detection2(image):
    # Apply edge detection to an image
    sobel_x = apply_kernel(image, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y = apply_kernel(image, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    return sobel_x, sobel_y

def image_segmentation(image, num_clusters):
    # Perform image segmentation on an image
    flattened_pixels = image.array.reshape(-1, image.num_channels)
    # Implement your own or use existing clustering algorithm (e.g., K-means)
    clusters = cluster_pixels(flattened_pixels, num_clusters)
    segmented_image = clusters.reshape(image.x_pixels, image.y_pixels, image.num_channels)
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
    new_im.array = segmented_image

    return new_im

def compress_image(image, compression_ratio):
    # Convert the image array to uint8 format
    image = img_as_ubyte(image)

    # Create a temporary file to save the compressed image
    temp_file = "temp.jpg"

    # Save the image with specified compression quality
    cv2.imwrite(temp_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), int((1 - compression_ratio) * 100)])

    # Read the compressed image back
    compressed_image = cv2.imread(temp_file)

    # Remove the temporary file
    os.remove(temp_file)

    return compressed_image


def image_compression(image, compression_ratio):
    # Apply image compression to reduce the file size while maintaining image quality
    compressed_image = compress_image(image.array, compression_ratio)

    # Create a new Image object with the compressed image data
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels, array=compressed_image)

    return new_im

def calculate_ssim(img1, img2):
    # Implementation of SSIM calculation
    # Compute the mean of the images
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    # Compute the standard deviation of the images
    std1 = np.std(img1)
    std2 = np.std(img2)

    # Compute the covariance of the images
    cov = np.cov(img1.flatten(), img2.flatten())[0, 1]

    # Compute the SSIM
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / ((mean1 ** 2 + mean2 ** 2 + c1) * (std1 ** 2 + std2 ** 2 + c2))

    return ssim


def image_comparison(image1, image2):
    # Compare two images and calculate a similarity metric
    img1 = np.clip(image1.array * 255, 0, 255).astype(np.uint8)
    img2 = np.clip(image2.array * 255, 0, 255).astype(np.uint8)

    # Resize the images if they have different dimensions
    if img1.shape != img2.shape:
        img2 = resize(img2, img1.shape)

    mse = np.mean((img1 - img2) ** 2)
    ssim = calculate_ssim(img1, img2)

    return mse, ssim

def cluster_pixels(image, num_clusters):
    # Perform image segmentation using K-means clustering
    pixels = image.reshape(-1, 3)  # Reshape image to a 2D array
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(pixels)
    segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)
    return segmented_image

def calculate_ssim(img1, img2):
    # Implementation of SSIM calculation
    # Compute the mean of the images
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    # Compute the standard deviation of the images
    std1 = np.std(img1)
    std2 = np.std(img2)

    # Compute the covariance of the images
    cov = np.cov(img1.flatten(), img2.flatten())[0, 1]

    # Compute the SSIM
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / ((mean1 ** 2 + mean2 ** 2 + c1) * (std1 ** 2 + std2 ** 2 + c2))

    return ssim


def plot_histogram(image, save_path=None):
    if image.num_channels == 1:
        # Grayscale image
        hist, bins = np.histogram(image.array.flatten(), bins=256, range=(0, 1))

        plt.figure()
        plt.plot(bins[:-1], hist, color='black')
        plt.title("Grayscale Image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    elif image.num_channels == 3:
        # Color image
        hist_red, bins_red = np.histogram(image.array[:, :, 0].flatten(), bins=256, range=(0, 1))
        hist_green, bins_green = np.histogram(image.array[:, :, 1].flatten(), bins=256, range=(0, 1))
        hist_blue, bins_blue = np.histogram(image.array[:, :, 2].flatten(), bins=256, range=(0, 1))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(bins_red[:-1], hist_red, color='red')
        plt.title("Red Channel")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 2)
        plt.plot(bins_green[:-1], hist_green, color='green')
        plt.title("Green Channel")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        plt.plot(bins_blue[:-1], hist_blue, color='blue')
        plt.title("Blue Channel")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    else:
        raise ValueError("Histogram visualization is only supported for grayscale and color images.")

    plt.tight_layout()

    if save_path:
        output_dir = image.output_path
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(save_path))[0]
        save_path = os.path.join(output_dir, file_name + '_histogram.png')
        plt.savefig(save_path)
    else:
        plt.show()

def image_blending(image1, image2, alpha):
    # Perform image blending by linearly combining two images based on alpha value
    # Resize the images to match if needed
    if image1.shape != image2.shape:
        new_width = min(image1.shape[0], image2.shape[0])
        new_height = min(image1.shape[1], image2.shape[1])
        image1 = image1[:new_width, :new_height, :]
        image2 = image2[:new_width, :new_height, :]
    blended_image = alpha * image1 + (1 - alpha) * image2
    return blended_image


def image_sharpening(image, alpha):
    # Perform image sharpening by enhancing high-frequency components using a Laplacian operator
    image=np.array(image)
    blurred_image = blur2(image, 3)
    sharpened_image = image + alpha * (image - blurred_image)
    return sharpened_image

def convert_to_grayscale(image):
    # Convert an image to grayscale by taking the average of RGB channels
    grayscale_image = np.mean(image, axis=2, keepdims=True)
    return grayscale_image

def apply_vignette_effect(image, alpha):
    # Apply a vignette effect to the image by darkening the corners
    x_pixels, y_pixels, _ = image.shape
    xx, yy = np.mgrid[:x_pixels, :y_pixels]
    distance = np.sqrt((xx - x_pixels / 2) ** 2 + (yy - y_pixels / 2) ** 2)
    distance = distance / np.max(distance)  # Normalize distance
    distance = distance.reshape(x_pixels, y_pixels, 1)  # Reshape distance to match image shape
    vignette_image = image * (1 - alpha * distance)
    return vignette_image


def blur2(image, kernel_size):
    # Apply a simple box blur to the image using a square kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    blurred_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[2]):
        blurred_channel = np.convolve(image[:, :, channel].flatten(), kernel.flatten(), mode='same')
        blurred_image[:, :, channel] = blurred_channel.reshape(image.shape[:2])
    return blurred_image
def image_drawing(image, x, y, color=(255, 0, 0), thickness=1, alpha=1.0):
    # Draw a line on the image
    drawn_image = Img.fromarray(image.array.astype(np.uint8))
    draw = ImgDraw.Draw(drawn_image)
    draw.line((x[0], y[0], x[1], y[1]), fill=color, width=thickness)
    blended_image = alpha * image.array + (1 - alpha) * np.array(drawn_image)
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels,
                   array=blended_image)
    return new_im

def pencil_sketch(image, alpha):
    # Convert the image to grayscale
    grayscale_image = np.dot(image.array[..., :3], [0.2989, 0.587, 0.114])
    # Invert the grayscale image
    inverted_image = 255 - grayscale_image
    # Apply Gaussian blur to the inverted image
    blurred_image = apply_gaussian_blur(inverted_image, sigma=21)
    # Blend the grayscale image and the blurred image using color dodge blending mode
    pencil_sketch = color_dodge(grayscale_image, blurred_image)
    pencil_sketch = np.expand_dims(pencil_sketch, axis=2)
    blended_image = alpha * image.array + (1 - alpha) * pencil_sketch
    new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels,
                   array=blended_image)
    return new_im


def apply_gaussian_blur(image_array, sigma):
    # Apply Gaussian blur to the image array
    blurred_image = Img.fromarray(image_array.astype(np.uint8)).filter(ImgFilter.GaussianBlur(radius=sigma))
    blurred_image_array = np.array(blurred_image)
    return blurred_image_array


def color_dodge(image1, image2):
    # Perform color dodge blending between two images
    result = np.divide(image1, 255 - image2, out=np.zeros_like(image1), where=(255 - image2) != 0)
    return np.clip(result * 255, 0, 255)

