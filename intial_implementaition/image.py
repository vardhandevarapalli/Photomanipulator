import numpy as np
import png
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
        '''
        read PNG RGB image, return 3D numpy array organized along Y, X, channel
        values are float, gamma is decoded
        '''
        im = png.Reader(self.input_path + filename).asFloat()
        resized_image = np.vstack(list(im[2]))
        resized_image.resize(im[1], im[0], 3)
        resized_image = resized_image ** gamma
        return resized_image
    
    def write_image(self, output_file_name, gamma=2.2):
        im = np.clip(self.array, 0, 1)
        im = np.nan_to_num(im)  # Replace NaN values with 0
        y, x = self.array.shape[0], self.array.shape[1]
        if self.num_channels == 1:
            im = im.reshape(y, x)
            writer = png.Writer(x, y, greyscale=True)
        else:
            im = im.reshape(y, x * self.num_channels)
            writer = png.Writer(x, y)
        with open(self.output_path + output_file_name, 'wb') as f:
            writer.write(f, (im * 255).astype(np.uint8))

        self.array = np.copy(self.array)  # Create a new copy of the array
        if self.num_channels == 1:
            self.array.resize(y, x)
        else:
            self.array.resize(y, x, self.num_channels)

if __name__ == '__main__':
    im = Image(filename='lake.png')
    im.write_image('test3.png')
