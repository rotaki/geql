from EncodeState import EncodeState
from DecodeState import DecodeState
from StateEncodingParams import StateEncodingParams
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

"""
* clustering the training sets
* editing action-state table
"""
class Cluster(EncodeState, DecodeState):
    def __init__(self, state_encoding_params, action_space_size, clustering_method, n_clusters):
        self.s_e_p = state_encoding_params
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters if n_clusters is not None else None
        self.action_counter = np.zeros((n_clusters, action_space_size))

    # Do clustering with the collected training states
    def cluster(self, training_states):
        if self.clustering_method == "kmeans":
            self.model = KMeans(self.n_clusters).fit(training_states)

    # Returns predicted cluster of a state
    def predict_state_cluster(self, state):
        return self.model.predict(self.encode_state(self.clustering_method, state, self.s_e_p).reshape(1, -1))[0]

    # Show action count table
    def show_action_count(self):
        return self.action_counter
    
    # Returns action count of a state (cluster)
    def action_count(self, state):
        return self.action_counter[self.predict_state_cluster(state)]

    def gibbs_action_count(self, state):
        temp = np.exp(-1*self.action_counter[self.predict_state_cluster(state)])
        temp = temp/np.sum(temp)
        return temp

    # cluster center
    def save_cluster_image(self):
        for i, imageArray in enumerate(self.model.cluster_centers_):
            cluster_img = self.decode_state(self.clustering_method, imageArray, self.s_e_p)
            cluster_img.save("./cluster_img_ds{}_pi{}/cluster_{}.png".format(self.s_e_p.resize_factor,self.s_e_p.pixel_intensity,i))
            # cluster_img.show()
        
    # Add one to the count when action taken from a state
    def add_action_count(self, state, action):
        self.action_counter[self.predict_state_cluster(state)][action] += 1
