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
        # If the state-action tuple isn't in the table, return 0
        return self.q_table.get((self.encode_state(state), action), 0.0) 

    def batch_estimate(self, state, actions):
        return map(lambda a: (a, self.estimate(state, a)), actions)
    
    def reward(self, state, action, reward, state2, action2):
        old_estimate = self.estimate(state, action)
        result_state_value = self.estimate(state2, action2) if action2 is not None else 0
        temporal_error = reward + self.discount * result_state_value - old_estimate
        new_estimate = old_estimate + self.learning_rate * temporal_error
        self.q_table[(self.encode_state(state), action)] = new_estimate

    def episode_finished(self):
        # Do nothing here for this QEstimator
        # (if we wanted to do per-episode updates, we'd do it here)
        pass

    # TODO: save/load
