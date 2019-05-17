from RLInterfaces import IActionPolicy
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from EncodeState import EncodeState
from DecodeState import DecodeState

class ClusterEpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon, state_encoding_params):
        self.epsilon = epsilon
        self.actions = actions
        self.s_e_p = state_encoding_params
        self.collected_pretraining_states = []
        self.cluster_model = None
        self.action_counter = np.zeros((self.s_e_p.n_clusters, len(self.actions)))
        self.n_episodes = 0
        self.n_steps = 0

    def encode_state(self, state):
        return EncodeState().encode_state(clustering_method="kmeans",
                                          state=state,
                                          state_encoding_params=self.s_e_p)

    def decode_state(self, encoded_state):
        return DecodeState().decode_state(clustering_method="kmeans",
                                          encoded_state=encoded_state,
                                          state_encoding_params=self.s_e_p)

    def cluster_center(self):
        return self.cluster_model.cluster_centers_

    def show_cluster_image(self):
        for i, imageArray in enumerate(self.cluster_model.cluster_centers_):
            cluster_img = self.decode_state(imageArray)
            cluster_img.show()
            
    # cluster center
    def save_cluster_image(self):
        # Remove cluster image folder if there and create a new one
        if os.path.exists("cluster_img_ds{}_pi{}".format(ds,pi)):
            import shutil
            shutil.rmtree("cluster_img_ds{}_pi{}".format(ds,pi))
        os.makedirs("./cluster_img_ds{}_pi{}".format(ds,pi))

        for i, imageArray in enumerate(self.cluster_model.cluster_centers_):
            cluster_img = self.decode_state(imageArray)
            cluster_img.save("./cluster_img_ds{}_pi{}/cluster_{}.png".format(self.s_e_p.resize_factor,self.s_e_p.pixel_intensity,i))


    def save_collected_states(self):
        ds = self.s_e_p.resize_factor
        pi = self.s_e_p.pixel_intensity

        # Compressed folder for storing states
        if not os.path.exists("pretraining_states_ds{}_pi{}.npz".format(ds,pi)):
            from pathlib import Path
            Path('pretraining_states_ds{}_pi{}.npz'.format(ds,pi)).touch()
            
            np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(ds,pi), [-1])

        self.saved_pretraining_states = np.load("./pretraining_states_ds{}_pi{}.npz".format(ds, pi))["arr_0"]
        if self.saved_pretraining_states.shape == (1,):
            self.saved_pretraining_states = np.array(self.collected_pretraining_states)
        else:
            self.saved_pretraining_states = np.concatenate([self.saved_pretraining_states, np.array(self.collected_pretraining_states)], 0)

        self.saved_pretraining_states = np.unique(self.saved_pretraining_states, axis=0)

        np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity), self.saved_pretraining_states)

        print("COLLECTED pretraining states are saved to SAVED pretraing states")
        print("Duplicated states are removed")

    def show_action_count(self):
        if self.action_counter is not None:
            return self.action_counter
        else:
            return

    # Returns action count of a cluster
    def action_count(self, cluster):
        return self.action_counter[cluster]

        # Add one to the count when action taken from a state
    def add_action_count(self, cluster, action):
        self.action_counter[cluster][action] += 1

    def collect_state(self,state):
        if self.n_steps  % self.s_e_p.s_c_i == 0:
             imgArray = self.encode_state(state)
             self.collected_pretraining_states.append(imgArray)
        return

    # Returns predicted cluster of a state
    def predict_state_cluster(self, state):
        return self.cluster_model.predict(self.encode_state(state).reshape(1, -1))[0]


    def gibbs_action_count(self, cluster):
        temp = np.exp(-1*self.action_counter[cluster])
        temp = temp/np.sum(temp)
        return temp
    
        
    def get_action(self, state, q_estimator):
        self.n_steps += 1
        self.collect_state(state)
        if random.random() < self.epsilon:
            # Choose randomly
            if self.cluster_model is not None:
                mask = self.gibbs_action_count(self.predict_state_cluster(state))
                action_choice = np.random.choice(self.actions, p=mask)
                self.add_action_count(self.predict_state_cluster(state), action_choice)
            else:
                action_choice = np.random.choice(self.actions)
            return action_choice
        
        else:
            # Choose greedily (break ties randomly)
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_v = max(action_values, key=lambda av: av[1])[1]
            candidates = list(filter(lambda av: av[1] == best_v, action_values))
            chosen = random.choice(candidates)
            if self.cluster_model is not None:
                self.add_action_count(self.predict_state_cluster(state), chosen[0])
            return chosen[0]

    def episode_finished(self):
        self.n_episodes += 1
        if self.n_episodes < self.s_e_p.batch_size:
            print('{}/{} episodes finished for next cluster'.format(self.n_episodes,
                                                                    self.s_e_p.batch_size))
            return
        self.n_episodes = 0
        print('{}/{} training kmeans...'.format(self.n_episodes, self.s_e_p.batch_size))

        if np.array(self.collected_pretraining_states).shape[0] >= self.s_e_p.n_clusters:
            self.cluster_model = KMeans(self.s_e_p.n_clusters).fit(self.collected_pretraining_states)
            self.collected_pretraining_states = []
            # If you want to save the cluster_center image or collected states
            # self.save_cluster_image() 
            # self.save_collected_states()
        else:
            print("number of unique samples for kmeans: too small. resume")

    def summary(self):
        return 'Cluster $\epsilon-greedy$ [$\epsilon = {}$, $|A| = {}$]'.format(self.epsilon, len(self.actions))

    # TODO: Load/save
