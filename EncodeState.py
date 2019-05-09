from PIL import Image
import numpy as np

"""
* encode state for training + predicting(States are optimized for different learning methods)
"""
class EncodeState:
    def __init__(self, resize_factor):
        self.resize_factor = 8 if resize_factor is None else resize_factor
        
    def encode_state(self, clustering_method, state):
        if (clustering_method == "kmeans"):
            img = Image.fromarray(state)
            img = img.crop((0,40,256,240))
            img = img.convert(mode='L')
            resized_img = img.resize((int(img.width/self.resize_factor), int(img.height/self.resize_factor)), resample=Image.BICUBIC)
            imgtoArray = np.asarray(resized_img).reshape(-1)
            return imgtoArray

