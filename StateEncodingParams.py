class StateEncodingParams:
    def __init__(self, default_shape, resize_factor, pixel_intensity):
        self.default_shape = (default_shape[0], default_shape[1]-40)
        self.resize_factor = resize_factor
        self.pixel_intensity = pixel_intensity
