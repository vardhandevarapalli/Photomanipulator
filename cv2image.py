import os
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, input_folder='input/', output_folder='output/'):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def load_image(self, file):
        image_path = os.path.join(self.input_folder, file)
        image = cv2.imread(image_path)
        return image

    def save_image(self, image, filename):
        output_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_path, image)