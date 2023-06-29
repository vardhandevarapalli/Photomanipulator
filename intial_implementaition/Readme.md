# Photo Manipulator using Python

This repository contains a Python library for photo manipulation tasks. It provides functionality for reading and writing PNG images, as well as implementing various image transformation functions. The library also includes a script for easy execution of these functions with user-defined inputs.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Execution](#execution)
- [Functions](#functions)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Photo Manipulator library offers a set of functions for performing various photo manipulation tasks. These tasks include histogram equalization, image compression, colorization, image comparison, brightness adjustment, contrast adjustment, blurring, edge detection, rotation, cropping, resizing, image segmentation, image drawing, image blending, image sharpening, grayscale conversion, and applying a vignette effect.

The library is implemented using the following main files:

- **png.py**: This file contains the PNG Reader and Writer classes, which provide functionality for reading and writing PNG images. These classes are based on the pure Python PNG Reader and Writer classes developed by Johann C. Rocholl.

- **image.py**: The *image.py* file includes the Image class, which utilizes the PNG Reader and Writer classes to read and write images. It provides a convenient interface for working with images and acts as a wrapper for the PNG Reader and Writer.

- **transform.py**: The *transform.py* file implements various image transformation functions. These functions are used for performing different photo manipulation tasks, such as histogram equalization, image compression, colorization, image comparison, brightness adjustment, contrast adjustment, blurring, edge detection, rotation, cropping, resizing, image segmentation, image drawing, image blending, image sharpening, grayscale conversion, and applying a vignette effect.

- **execute.py**: The *execute.py* script allows easy execution of the image transformation functions provided in *transform.py*. It prompts the user to input the required parameters for a chosen function and then executes the function accordingly.

- **tests.py**: The *tests.py* file contains test cases for all the functions in the *transform.py* module. These tests ensure the correct functioning of the photo manipulation functions.

## File Structure

The repository is structured as follows:

photo-manipulator```/
├── png.py
├── image.py
├── transform.py
├── execute.py
└── tests.py```

## Execution

To use the Photo Manipulator library, follow these steps:

1. Clone the repository: ```git clone https://github.com/your-username/photo-manipulator.git```

2. Ensure that you have Python 3.x installed on your system.

3. Install the required dependencies using pip: ```pip install pillow```

4. Execute the `execute.py` script: ```python execute.py```

This script allows you to interactively choose a function from the available options and input the required parameters. The script then executes the chosen function and generates the manipulated photo as output.

## Functions

The Photo Manipulator library provides the following functions in the `transform` module:

- `plot_histogram`: Plots the histogram of an image.
- `image_compression`: Compresses an image based on a given compression factor.
- `colorize`: Colorizes an image using a specified color palette.
- `image_comparison`: Compares two images using the structural similarity index (SSIM).
- `brighten`: Adjusts the brightness of an image.
- `adjust_contrast`: Adjusts the contrast of an image.
- `blur`: Applies blurring to an image using a specified kernel size.
- `apply_kernel`: Applies a custom kernel to an image.
- `rotate`: Rotates an image by a specified angle.
- `crop`: Crops an image to a specified region of interest.
- `resize2`: Resizes an image to a specified width and height.
- `edge_detection2`: Performs edge detection on an image using the Sobel operator.
- `image_segmentation`: Segments an image into a specified number of regions.
- `image_drawing`: Draws shapes or lines on an image.
- `image_blending`: Blends two images together using a specified alpha value.
- `image_sharpening`: Sharpens an image using a specified alpha value.
- `convert_to_grayscale`: Converts an image to grayscale.
- `apply_vignette_effect`: Applies a vignette effect to an image.

Refer to the source code and function docstrings for more details on the usage of each function.

## Tests

The `tests.py` file contains test cases for all the functions in the `transform` module. These tests ensure the correct behavior and accuracy of the photo manipulation functions. To run the tests, execute the following command: ```python tests.py```








