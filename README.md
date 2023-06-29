# Photomanipulator
This repository holds the codes for the Photomanipulator using python

# Dual Implementation Project

This project showcases two different implementations of an Image Manipulator using different tools. The first implementation utilizes the files `png.py`, `image.py`, `transform.py`, `tests.py`, and `execute.py`, while the second implementation uses the source code in the files `cv2image.py`, `cv2transform.py`, and `cv2execute.py`, primarily relying on the `OpenCV` library.

## Table of Contents

- [Introduction](#introduction)
- [Implementation 1: Image Manipulator](#implementation-1-image-manipulator)
- [Implementation 2: OpenCV](#implementation-2-opencv)
- [Sub-Implementation: PIL Image](#sub-implementation-pil-image)
- [Usage](#usage)

## Introduction

The Dual Implementation Project provides two separate implementations of an Image Manipulator, demonstrating different tools and approaches. The first implementation utilizes a combination of Python files in the `intial_implementation` directory, namely `png.py`, `image.py`, `transform.py`, `tests.py`, and `execute.py`, to perform various image processing operations. The second implementation relies primarily on the `OpenCV` library, with the source code distributed across `cv2image.py`, `cv2transform.py`, and `cv2execute.py` in the `Opencvimplementation`.

## Implementation 1: Image Manipulator

The first implementation of the Image Manipulator utilizes a combination of Python files, namely `png.py`, `image.py`, `transform.py`, `tests.py`, and `execute.py`. These files contain the necessary code for histogram equalization, image compression, colourization, edge detection, and more.

## Implementation 2: OpenCV

The second implementation of the Image Manipulator heavily relies on the OpenCV library. The source code is distributed across the files `cv2image.py`, `cv2transform.py`, and `cv2execute.py`. `OpenCV` provides a wide range of functions and algorithms for image processing, including image manipulation, filtering, edge detection, and feature extraction. This implementation leverages the power and versatility of OpenCV for performing advanced image processing tasks.

## Sub-Implementation: PIL Image

Within the first implementation, the `image.py` file presents an alternative option for image extraction using the PIL library. Users can use the `PIL` library instead of the default PNG library for image extraction by modifying the code. This sub-implementation file, `imagealt.py`, in the `intial_implementation` directory provides the necessary code to extract images using the `PIL` library.

## Usage

To use either implementation, refer to the specific files and follow the provided instructions within each implementation directory. Detailed usage instructions and code examples can be found in the respective directories.

1. Refer to the directory `intial_implentation` for detailed usage instructions and code examples for the Image Manipulator implementation.

2. For the OpenCV implementation, refer to the directory `Opencvimplementation` for detailed usage instructions and code examples utilizing the OpenCV library.

In the case of the Image Manipulator implementation, if you wish to use the PIL library for image extraction, you can find the necessary code in the `imagealt.py` file in the `intial_implementation` directory.

### Note
used `png.py:` pure Python PNG `Reader` and `Writer` classes from Johann C. Rocholl
