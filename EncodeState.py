from PIL import Image
from PIL import ImageFilter
import numpy as np

"""
* encode state for training + predicting(States are optimized for different learning methods)
"""
class EncodeState:
    def digitize_state(self, state_encoding_params):
        return np.linspace(0, 256, state_encoding_params.pixel_intensity + 1)[:-1]
    
    def encode_state(self, clustering_method, state, state_encoding_params):
        if (clustering_method == "kmeans"):
            img = Image.fromarray(state)
            img = img.crop((0,40,256,240))
            img = img.convert(mode='L')
            img = img.filter(ImageFilter.FIND_EDGES)
            
            new_width = state_encoding_params.default_shape[1]//state_encoding_params.resize_factor
            new_height = state_encoding_params.default_shape[0]//state_encoding_params.resize_factor
            resized_img = img.resize((new_width, new_height), resample=Image.BICUBIC)
            imgtoArray = np.asarray(resized_img).reshape(-1)
            imgtoArray = np.digitize(imgtoArray, bins=self.digitize_state(state_encoding_params))*(256/state_encoding_params.pixel_intensity)
            return imgtoArray
        elif (clustering_method == "agressive_ds"):
            img = Image.fromarray(state)
            img = img.crop((0,40,256,240))
            img = img.convert(mode='L')
            img = img.filter(ImageFilter.FIND_EDGES)
            
            new_width = state_encoding_params.default_shape[1]//state_encoding_params.resize_factor
            new_height = state_encoding_params.default_shape[0]//state_encoding_params.resize_factor
            resized_img = img.resize((new_width, new_height), resample=Image.BICUBIC)
            imgtoArray = np.asarray(resized_img).reshape(-1)
            imgtoArray = np.digitize(imgtoArray, bins=self.digitize_state(state_encoding_params))*(256/state_encoding_params.pixel_intensity)
            return zlib.compress(imgtoArray.tobytes(), state_encoding_params.compression)


