from encode import EncodeState
import numpy as np

"""
* collect training samples and store it in a table
"""
class TrainingStates(EncodeState):
    def __init__(self, env, clustering_method, steps):
        super().__init__(resize_factor=None)
        self.training_states = []
        self.steps = steps
        self.clustering_method = clustering_method
        self.env = env


    # Returns training states with encoding
    def get_training_states(self):
        done = True
        for x in range(self.steps):
            if done:
                state = self.env.reset()
                
            # Control action somehow
            action = self.env.action_space.sample()
            
            state, reward, done, info = self.env.step(action)
            print(x)
            self.training_states.append(self.encode_state(self.clustering_method, state))
        return np.array(self.training_states)
