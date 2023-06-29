import numpy as np
from PIL import Image as PILImage


class Image:
    def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, filename='', array=None):
        self.input_path = 'input/'
        self.output_path = 'output/'
        if array is not None:
            self.array = array
            shape = self.array.shape
            if len(shape) == 2:
                self.x_pixels, self.y_pixels = shape[:2]
                self.num_channels = 1
            else:
                self.x_pixels, self.y_pixels, self.num_channels = shape[:3]
        elif x_pixels and y_pixels and num_channels:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
            self.num_channels = num_channels
            self.array = np.empty((x_pixels, y_pixels, num_channels), dtype=np.float32)
        elif filename:
            self.array = self.read_image(filename)
            shape = self.array.shape
            if len(shape) == 2:
                self.x_pixels, self.y_pixels = shape[:2]
                self.num_channels = 1
            else:
                self.x_pixels, self.y_pixels, self.num_channels = shape[:3]
        else:
            raise ValueError("You need to input either a filename OR specify the dimensions of the image")

    def read_image(self, filename, gamma=2.2):
        """
        Read PNG RGB image, return 3D numpy array organized along Y, X, channel.
        Values are float, gamma is decoded.
        """
        image = PILImage.open(self.input_path + filename)
        image = image.convert('RGB')
        resized_image = np.array(image.resize((image.width, image.height)))
        resized_image = resized_image.astype(np.float32) / 255.0
        resized_image = resized_image ** gamma
        return resized_image

    def write_image(self, output_file_name, gamma=2.2):
        """
        Write image to a PNG file.
        Gamma encode the values, clip and convert to 8-bit integers.
        """
        output_array = np.clip(self.array, 0, 1) ** (1 / gamma)
        output_array = (output_array * 255).astype(np.uint8)

        if self.num_channels == 1:
            output_image = PILImage.fromarray(output_array.reshape(self.y_pixels, self.x_pixels))
            output_image = output_image.convert('L')
        else:
            output_image = PILImage.fromarray(output_array)

        output_image.save(self.output_path + output_file_name)

        self.array = np.copy(self.array)
        if self.num_channels == 1:
            self.array.resize(self.y_pixels, self.x_pixels)
        else:
            self.array.resize(self.y_pixels, self.x_pixels, self.num_channels)


if __name__ == '__main__':
    im = Image(filename='towere.png')
    im.write_image('testalt.png')
