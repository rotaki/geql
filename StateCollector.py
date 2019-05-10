from EncodeState import EncodeState
import numpy as np
import getch
import sys

"""
* collect training samples and store it in a table
"""
class TrainingAgent(EncodeState):
    def __init__(self, environment, clustering_method, n_training_steps, action_interval, sample_collect_interval, state_encoding_params):
        self.env = environment
        self.clustering_method = clustering_method
        self.current_episode = 0
        self.training_states = []
        self.steps = n_training_steps
        self.frames = 0
        self.action_interval = action_interval
        self.s_c_i = sample_collect_interval
        self.s_e_p = state_encoding_params

    def action_choice(self):
        # TODO: fix for two characters e.g) 10, 11
        x = ord(getch.getch())

        return x


    # Returns training states with encoding
    def get_training_states(self):
        print("===========================================================================================")
        print("Pretraining session! Let Mario explore as much as possibile!!")
        print("Arrows to play basic moves. Space is jump forward (right+A+B)")
        print("Otherwise, 0 to {} to play, Press q to quit.".format(self.env.action_space.n-1))
        print("===========================================================================================")
        
        done = True
        for x in range(self.steps):
            if done:
                state = self.env.reset()
                
            # Control action from keyboard
            while (True):
                key = self.action_choice()
                if (key == 65): # arrow-up
                    action = 5  # jump
                elif (key == 66): # arrow-down
                    action = 10   # down
                elif (key == 67): # arrow-right
                    action = 3    # right+B
                elif (key == 68): # arrow-left
                    action = 8    # left+B
                elif (key == 32): # space
                    action = 4    # right+A+B
                else:
                    action = key - 48
                if action in np.arange(self.env.action_space.n):
                    break
                # Press q to quit
                elif action == 113-48:
                    sys.exit()
            
            for frame in range(self.action_interval):
                next_state, reward, done, info = self.env.step(action)
                if done:
                    break

            if x % self.s_c_i == 0:
                self.training_states.append(self.encode_state(self.clustering_method,
                                                              next_state,
                                                              self.s_e_p))

            self.env.render()
            
        return np.array(self.training_states)
