# Photo Manipulator Toolbox using OpenCV

The Photo Manipulator Toolbox is a Python library that provides a collection of image-processing functions and tools using the OpenCV library. It aims to simplify manipulating images by offering a range of ready-to-use functions that can be easily integrated into your projects.

## Features

**Comprehensive Image Processing Functions:** The toolbox offers various image processing functions, including contrast adjustment, blurring, edge detection, image blending, colour manipulation, image compression, resizing, rotation, and more. These functions are implemented using popular libraries such as OpenCV and NumPy to ensure efficiency and accuracy.

**User-Friendly Interface:** The toolbox provides a user-friendly interface for executing image processing functions. It includes an `execute_image_function` function that easily applies any desired function to an input image. The function prompts you to provide values for the function's parameters and automatically handles the loading and saving of images.

**Flexible Parameter Handling:** The toolbox leverages Python's `inspect` module to retrieve the parameter information of each image processing function dynamically. This enables seamless interaction with the functions, allowing you to provide input values for various parameter types, including numerical arrays, lists, tuples, and more.

**Image Processing Class:** The toolbox includes an `ImageProcessor` class that simplifies the loading and saving of images. It provides convenient methods for loading an image from a file, saving it to a specific location, and handling the conversion and scaling of image data.

## Installation

To use the Image Processing Toolbox, follow these steps:

1. Clone the repository or download the source code.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Import the toolbox modules into your Python project.

## Execution

To execute the Image Processing Toolbox and apply image processing functions, you can use the `cv2execute.py` file. This file provides an interface for selecting and executing the desired image processing function. Here's how to use it:

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command: `python cv2execute.py`.
4. Follow the instructions in the command prompt to select and execute an image processing function.

## Available Image Manipulating Functions

The Image Processing Toolbox provides the following image-manipulating functions:

- `adjust_contrast`: Adjusts the contrast of an image.
- `apply_gaussian_blur`: Applies Gaussian blur to an image.
- `apply_kernel`: Applies a custom kernel to an image.
- `apply_vignette_effect`: Applies a vignette effect to an image.
- `blur`: Blurs an image using the average blur technique.
- `blur2`: Blurs an image using the median blur technique.
- `brighten`: Brightens an image by a specified factor.
- `calculate_ssim`: Calculates the structural similarity index (SSIM) between two images.
- `cluster_pixels`: Segments an image by clustering its pixels.
- `color_dodge`: Blends two images using the colour dodge blending mode.
- `colorize`: Colorizes a grayscale image using a specified colour palette.
- `compress_image`: Compresses an image using a specified compression factor.
- `convert_to_grayscale`: Converts an image to grayscale.
- `crop`: Crops an image to a specified region of interest.
- `edge_detection_x`: Detects edges in an image using the Sobel operator along the x-axis.
- `edge_detection_y`: Detects edges in an image using the Sobel operator along the y-axis.
- `image_blending`: Blend two images using a specified alpha value.
- `image_comparison`: Compares two images based on mean squared error (MSE) and structural similarity index (SSIM).
- `image_compression`: Compresses and decompresses an image using JPEG encoding.
- `image_drawing`: Draws lines on an image with specified coordinates, colour, thickness, and blending alpha.
- `image_segmentation`: Segments an image by clustering its pixels.
- `image_sharpening`: Sharpens an image by applying a Gaussian blur and blending with the original image.
- `img_as_ubyte`: Converts an image to 8-bit unsigned integer format.
- `pencil_sketch`: Converts an image to a pencil sketch.
- `plot_histogram`: Plots the histogram of an image.
- `resize`: Resizes an image to a specified width and height.
- `resize2`: Resizes an image to a specified width and height using interpolation.
- `rotate`: Rotates an image by a specified angle

For more detailed information about each function, including the parameters and usage examples, please look at the documentation or docstrings provided in the code in the files `cv2image.py`, `cv2transform.py` and `cv2execute.py`.
