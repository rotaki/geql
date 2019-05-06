from RLInterfaces import IQEstimator
import numpy as np
import zlib
from PIL import Image

import time

class TabularQEstimator (IQEstimator):
    def __init__(self, discount, learning_rate):
        self.q_table = dict()
        self.discount = discount
        self.learning_rate = learning_rate
        self.compression = 6
        self.downsampling = 8
        self.hits = 0
        self.misses = 0

    def summary(self):
        return 'TabQ [$\\alpha={}$, $\\gamma={}$]'.format(self.learning_rate,
                                                          self.discount)
        
    def encode_state(self, state):
        # This should probably be refactored into its own class, to make the
        # Q-estimator general for all problems
        
        i = Image.fromarray(state)

        # Crop to the ROI
        i = i.crop((0, 40, 256, 240))
        
        # Convert to grayscale
        i = i.convert(mode='L')
        
        # Downsample
        width, height = i.size
        i = i.resize((round(width / self.downsampling), round(height / self.downsampling)),
                     resample=Image.NEAREST)
        
        # Compress what's left
        return zlib.compress(i.tobytes(), self.compression)

    def estimate(self, state, action):
        encoded_state = self.encode_state(state)
        return self.estimate_encoded(encoded_state, action)

    def estimate_encoded(self, encoded_state, action):
        sa_tuple = (encoded_state, action)
        if sa_tuple in self.q_table:
            self.hits += 1
            return self.q_table[sa_tuple]
        else:
            self.misses += 1
            return 0.0
        
    def batch_estimate(self, state, actions):
        encoded_state = self.encode_state(state)
        return list(map(
            lambda a: (a, self.estimate_encoded(encoded_state, a)),
            actions))
    
    def reward(self, state, action, reward, state2, action2):
        V_sa = self.estimate(state, action)
        V_sa_next = self.estimate(state2, action2) if action2 is not None else 0
        V_sa_updated = V_sa + self.learning_rate * (reward + (self.discount * V_sa_next) - V_sa)
        self.q_table[(self.encode_state(state), action)] = V_sa_updated

    def episode_finished(self):
        queries = self.hits + self.misses
        print('TabQ hit ratio: {:.2f} % ({} hits, {} misses)'.
              format(self.hits/queries * 100, self.hits, self.misses))
        self.hits = 0
        self.misses = 0

    # TODO: save/load
