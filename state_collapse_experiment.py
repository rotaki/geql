import sys
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from PIL import Image
import numpy as np
# from sklearn.cluster import KMeans 

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


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

    
"""
* clustering the training sets
* editing action-state table
"""
class Cluster(EncodeState):
    def __init__(self, clustering_method, n_clusters):
        super().__init__(resize_factor=None)
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters if n_clusters is not None else None
        self.action_counter = np.zeros((n_clusters, env.action_space.n))

    # Do clustering with the collected training states
    def cluster(self, training_states):
        if self.clustering_method == "kmeans":
            from sklearn.cluster import KMeans
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


"""
* collect training samples and store it in a table
"""
class TrainingStates(EncodeState):
    def __init__(self, clustering_method, steps):
        super().__init__(resize_factor=None)
        self.training_states = []
        self.steps = steps
        self.clustering_method = clustering_method


    # Returns training states with encoding
    def get_training_states(self):
        done = True
        for x in range(self.steps):
            if done:
                state = env.reset()
                
            # Control action somehow
            action = env.action_space.sample()

            state, reward, done, info = env.step(action)
            
            self.training_states.append(self.encode_state(self.clustering_method, state))
        return np.array(self.training_states)


if __name__ == "__main__":
    clustering_method = "kmeans"    
    T = TrainingStates(clustering_method, 100)
    C = Cluster(clustering_method, 15)
    C.cluster(T.get_training_states())
    print(C.show_action_count())
    # Example new state and new action

    state = env.reset()
    action = env.action_space.sample()
    cluster = C.predict_state_cluster(state)
    print(cluster)
    C.add_action_count(state, action)
    print(C.show_action_count())
    print(C.gibbs_action_count(state))
