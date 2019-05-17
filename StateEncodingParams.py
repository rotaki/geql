class StateEncodingParams:
    def __init__(self, resize_factor, pixel_intensity, n_clusters=40,default_shape=(240, 256), compression=8, sample_collect_interval=4, batch_size=64):
        
        self.default_shape = (default_shape[0]-40, default_shape[1]) # 200, 256
        self.resize_factor = resize_factor
        self.pixel_intensity = pixel_intensity
        self.pixel_block = 256//pixel_intensity
        self.final_shape = ((default_shape[0]-40)//resize_factor,default_shape[1]//resize_factor)
        self.final_size = self.final_shape[0] * self.final_shape[1]
        self.compression = compression
        self.n_clusters = n_clusters
        self.s_c_i = sample_collect_interval
        self.batch_size = batch_size


