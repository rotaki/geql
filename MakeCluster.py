from EncodeState import EncodeState
from DecodeState import DecodeState
from StateEncodingParams import StateEncodingParams
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans

"""
Make cluster by collecting states and kmeans it
"""

class MakeCluster(EncodeState, DecodeState):
    def __init__(self,
                 state_encoding_params,
                 clustering_method,
                 n_clusters,
                 sample_collect_interval):
        
        self.s_e_p = state_encoding_params
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.s_c_i = sample_collect_interval
        self.cluster_model = 0
        
        ds = self.s_e_p.resize_factor
        pi = self.s_e_p.pixel_intensity
        
        # Compressed folder for storing states
        if not os.path.exists("pretraining_states_ds{}_pi{}.npz".format(ds,pi)):
            from pathlib import Path
            Path('pretraining_states_ds{}_pi{}.npz'.format(ds,pi)).touch()
            
            np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(ds,pi), [-1])

        # Remove cluster image folder if there and create a new one
        if os.path.exists("cluster_img_ds{}_pi{}".format(ds,pi)):
            import shutil
            shutil.rmtree("cluster_img_ds{}_pi{}".format(ds,pi))
        os.makedirs("./cluster_img_ds{}_pi{}".format(ds,pi))

        self.saved_pretraining_states = np.load("./pretraining_states_ds{}_pi{}.npz".format(ds, pi))["arr_0"]
        self.collected_pretraining_states = []
        self.cluster_model = 0

        
    # Do clustering with the collected training states
    def cluster(self, training_states):
        if self.clustering_method == "kmeans":
            self.cluster_model = KMeans(self.n_clusters).fit(training_states)

    # Returns predicted cluster of a state
    def predict_state_cluster(self, state):
        return self.cluster_model.predict(self.encode_state(self.clustering_method, state, self.s_e_p).reshape(1, -1))[0]


    def cluster_centers(self):
        return self.cluster_model.cluster_centers_

    def show_cluster_image(self):
        for i, imageArray in enumerate(self.cluster_model.cluster_centers_):
            cluster_img = self.decode_state(self.clustering_method, imageArray, self.s_e_p)
            cluster_img.show()
            
    # cluster center
    def save_cluster_image(self):
        for i, imageArray in enumerate(self.cluster_model.cluster_centers_):
            cluster_img = self.decode_state(self.clustering_method, imageArray, self.s_e_p)
            cluster_img.save("./cluster_img_ds{}_pi{}/cluster_{}.png".format(self.s_e_p.resize_factor,self.s_e_p.pixel_intensity,i))


    def collect_state(self,state,n_steps):
        if n_steps  % self.s_c_i == 0:
             imgArray = self.encode_state(self.clustering_method,
                                          state,
                                          self.s_e_p)
             self.collected_pretraining_states.append(imgArray)
        return

    def save_collected_states(self):
        if self.saved_pretraining_states.shape == (1,):
            self.saved_pretraining_states = np.array(self.collected_pretraining_states)
        else:
            self.saved_pretraining_states = np.concatenate([self.saved_pretraining_states, np.array(self.collected_pretraining_states)], 0)

        self.saved_pretraining_states = np.unique(self.saved_pretraining_states, axis=0)

        np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity), self.saved_pretraining_states)

        print("COLLECTED pretraining states are saved to SAVED pretraing states")
        print("Duplicated states are removed")

    def kmeans(self, current_episode, batch_size):
        if current_episode % batch_size == 0 and current_episode != 0:
            if np.array(self.collected_pretraining_states).shape[0] >= self.n_clusters:
                self.cluster(self.collected_pretraining_states)
                # If you want to save the cluster_center image
                # self.save_cluster_image()
                self.save_collected_states()
                return 1
            else:
                print("number of unique samples: too small. resume")
                return 0
        else:
            return 0
            
        


