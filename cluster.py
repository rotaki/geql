from encode import EncodeState
import numpy as np
from sklearn.cluster import KMeans

"""
* clustering the training sets
* editing action-state table
"""
class Cluster(EncodeState):
    def __init__(self, action_space_size, clustering_method, n_clusters):
        super().__init__(resize_factor=None)
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters if n_clusters is not None else None
        self.action_counter = np.zeros((n_clusters, action_space_size))

    # Do clustering with the collected training states
    def cluster(self, training_states):
        if self.clustering_method == "kmeans":
            self.model = KMeans(self.n_clusters).fit(training_states)

    # Returns predicted cluster of a state
    def predict_state_cluster(self, state):
        return self.model.predict(self.encode_state(self.clustering_method, state).reshape(1, -1))[0]

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
    
    # Add one to the count when action taken from a state
    def add_action_count(self, state, action):
        self.action_counter[self.predict_state_cluster(state)][action] += 1
