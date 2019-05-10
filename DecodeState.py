from PIL import Image
import numpy as np

"""
* decode state for training + predicting(States are optimized for different learning methods)
"""
class DecodeState:        
    def decode_state(self, clustering_method, encoded_state, state_encoding_params):
        if (clustering_method == "kmeans"):
            new_width = int(state_encoding_params.default_shape[0]/state_encoding_params.resize_factor)
            new_height = int(state_encoding_params.default_shape[1]/state_encoding_params.resize_factor)            
            imgArray = encoded_state.reshape(new_height, new_width)
            img = Image.fromarray(np.uint8(imgArray))
            return img
           
