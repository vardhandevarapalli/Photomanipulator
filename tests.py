from image import Image
#from imagealt import Image
import transform as tr
import numpy as np
if __name__ == '__main__':

    lake = Image(filename='lake.png')
    city = Image(filename='city.png')


    # Histogram Equalization
    im = Image(filename='lake.png')
    tr.plot_histogram(im, save_path='lake.png')

    # Image Compression
    compressed_im = tr.image_compression(lake, 0.8)
    compressed_im.write_image('compressed.png')

    # Colorize
    color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Example color palette
    colorized_im = tr.colorize(lake, color_palette)
    colorized_im.write_image('colorized.png')

    # Image Comparison
    mse, ssim = tr.image_comparison(lake, city)
    print(f"MSE: {mse}, SSIM: {ssim}")

    # brightening
    brightened_im = tr.brighten(lake, 1.7)
    brightened_im.write_image('brightened.png')

    # darkening
    darkened_im = tr.brighten(lake, 0.3)
    darkened_im.write_image('darkened.png')

    # increase contrast
    incr_contrast = tr.adjust_contrast(lake, 2, 0.5)
    incr_contrast.write_image('increased_contrast.png')

    # decrease contrast
    decr_contrast = tr.adjust_contrast(lake, 0.5, 0.5)
    decr_contrast.write_image('decreased_contrast.png')

    # blur using kernel 3
    blur_3 = tr.blur(city, 3)
    blur_3.write_image('blur_k3.png')

    # blur using kernel size of 15
    blur_15 = tr.blur(city, 15)
    blur_15.write_image('blur_k15.png')

    # let's apply a sobel edge detection kernel on the x and y axis
    sobel_x = tr.apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    sobel_x.write_image('edge_x.png')
    sobel_y = tr.apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y.write_image('edge_y.png')

    # Rotate
    rotated_im = tr.rotate(lake, 45)
    rotated_im.write_image('rotated.png')

    # Crop
    cropped_im = tr.crop(city, 100, 100, 200, 200)
    cropped_im.write_image('cropped.png')

    # Resize
    resized_im = tr.resize2(city, 400, 400)
    resized_im.write_image('resized.png')

    # Edge Detection
    edge_x, edge_y = tr.edge_detection2(city)
    edge_x.write_image('edge_x2.png')
    edge_y.write_image('edge_y2.png')

    # Image Segmentation
    segmented_im = tr.image_segmentation(city, 4)
    segmented_im.write_image('segmented.png')

    # Image drawing
    drawn_im = tr.image_drawing(lake, x=[100, 300], y=[200, 400], color=(255, 0, 0), thickness=2, alpha=0.8)
    drawn_im.write_image('drawn_image.png')
    # Perform image blending
    alpha = 0.5
    blended_image = Image(array=tr.image_blending(lake.array, city.array, alpha))
    blended_image.write_image('blended.png')

    # Perform image sharpening
    alpha = 0.7
    sharpened_image = Image(array=tr.image_sharpening(lake.array, alpha))
    sharpened_image.write_image('sharpened.png')

    # Perform grayscale conversion
    grayscale_image = Image(array=tr.convert_to_grayscale(lake.array))
    grayscale_image.write_image('grayscale.png')
    # Perform vignette effect

    alpha = 0.7
    vignette_image = Image(array=tr.apply_vignette_effect(lake.array, alpha))
    vignette_image.write_image('vignetted.png')