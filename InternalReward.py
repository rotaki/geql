import numpy as np

class InternalReward:
    def __init__(self):
        self.c = None
        
    def initialize_cluster_model(self, cluster_model):
        self.c = cluster_model
        self.bins =  np.linspace(0, self.c.s_e_p.final_size, 30+1)

    def internal_reward(self,state):
        if self.c is not None:
            imgArray = self.c.encode_state(self.c.clustering_method,
                                            state,
                                            self.c.s_e_p)
            prediction = self.c.predict_state_cluster(state)
            A = (imgArray//self.c.s_e_p.pixel_block)
            B = ((self.c.cluster_centers())[prediction]//self.c.s_e_p.pixel_block)
            internal_reward = np.sum(np.square(A-B))/(self.c.s_e_p.pixel_intensity**2)
            return internal_reward
        else:
            return 0
        
