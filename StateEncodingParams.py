class StateEncodingParams:
    def __init__(self, default_shape, resize_factor, pixel_intensity, compression=6):

        self.default_shape = (default_shape[0]-40, default_shape[1]) # 200, 256
        self.resize_factor = resize_factor
        self.pixel_intensity = pixel_intensity
        self.pixel_block = 256//pixel_intensity
        self.final_shape = ((default_shape[0]-40)//resize_factor,default_shape[1]//resize_factor)
        self.final_size = self.final_shape[0] * self.final_shape[1]
        self.compression = compression

