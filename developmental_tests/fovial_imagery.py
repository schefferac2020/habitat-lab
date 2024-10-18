import numpy as np
import cv2
from utils import *

class FovialImageFactory:
    def __init__(self, fovia_size, fovial_layers):
        self.image = np.zeros((512, 512, 3))
        self.fovia_size = fovia_size
        self.fovial_layers = fovial_layers
        self.fovia_layer_sizes = fovia_size*(2**np.arange(fovial_layers))
    
    def __init__(self, image, fovia_size, fovial_layers):
        self.image = image
        self.fovia_size = fovia_size
        self.fovial_layers = fovial_layers
        self.fovia_layer_sizes = fovia_size*(2**np.arange(fovial_layers))
    
    def set_image(self, image):
        self.image = image

    def get_fovial_image(self, x, y):
        if self.image is None:
            raise "Image parameter cannot be none..."
        
        fovial_stack_arr = self.__create_fovial_stack(self.image, x, y)
        return self.__make_fovial_img(fovial_stack_arr)

    # Private Functions
    def __create_fovial_stack(self, img, x, y):
        fovial_stack = np.zeros((self.fovia_size, self.fovia_size, self.fovial_layers*3), dtype=np.uint8) # 3 each for rgb


        half_fovia = int(self.fovia_size/2)

        x = int(np.clip(x, half_fovia, self.image.shape[1] - half_fovia))
        y = int(np.clip(y, half_fovia, self.image.shape[0] - half_fovia))

        for i, crop_width in enumerate(self.fovia_layer_sizes):
            cropped_img = crop_around_pixel(self.image, x, y, crop_width, crop_width)

            cropped_img = cv2.resize(cropped_img, (self.fovia_size, self.fovia_size), interpolation=cv2.INTER_LINEAR) #TODO: Is this the best way to squash the image down?
            fovial_stack[:, :, 3*i:3*i+3] = cropped_img


        return fovial_stack

    def __make_fovial_img(self, fovial_stack):
        fovial_image = np.zeros((self.fovia_layer_sizes[-1], self.fovia_layer_sizes[-1], 3), dtype=fovial_stack.dtype)

        for i, crop_width in reversed(list(enumerate(self.fovia_layer_sizes))):
            downsampled_img = fovial_stack[:, :, 3*i:3*i+3]
            downsampled_img = cv2.resize(downsampled_img, (crop_width, crop_width), interpolation=cv2.INTER_LINEAR)

            fovial_image = center_image(fovial_image, downsampled_img)

        return fovial_image