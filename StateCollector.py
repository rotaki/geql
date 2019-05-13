from EncodeState import EncodeState
import numpy as np
from MakeCluster import Cluster
import getch
import signal
from MarioRLAgent import *

"""
* collect training samples and store it in a table
"""
class PretrainingAgent(EncodeState, MarioRLAgent):
    def __init__(self,
                 environment,
                 q_estimator,
                 action_policy,
                 action_set,
                 action_interval,
                 clustering_method,
                 n_clusters,
                 sample_collect_interval,
                 state_encoding_params):
        
        MarioRLAgent.__init__(self,
                              environment,
                              q_estimator,
                              action_policy,
                              action_set,
                              learning_policy=LearningPolicy.Q,
                              action_interval=action_interval,
                              listener=None)
        
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.steps = 0
        self.s_c_i = sample_collect_interval
        self.s_e_p = state_encoding_params
        self.existing_pretraining_states = np.load("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity))["arr_0"]
        self.collected_pretraining_states = []
        self.done = True
        self.paused = True

        signal.signal(signal.SIGINT, self.make_signal_handler())

    def make_signal_handler(self):
        def handler(signum, frame):
            if not self.paused:
                print('Ctrl-C caught! Pausing...')
                self.paused = True
        return handler

    def action_choice(self):
        x = ord(getch.getch())
        return x


    def save_pretraining_states(self):
        self.show_states_status()
        print("Do you want to save the your collected pretraining states? (y/n)")
        while(True):
            comfirm_key_2 = getch.getch()
            if comfirm_key_2 == 'y':
                if len(self.collected_pretraining_states) == 0:
                    print("No collected pretraining states")
                    print("Proceed with the game")

                else:
                    if self.existing_pretraining_states.shape == (1,):
                        self.existing_pretraining_states = np.array(self.collected_pretraining_states)
                    else:
                        self.existing_pretraining_states = np.concatenate([self.existing_pretraining_states, np.array(self.collected_pretraining_states)], 0)

                    self.existing_pretraining_states = np.unique(self.existing_pretraining_states, axis=0)

                    np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity), self.existing_pretraining_states)
                    print("COLLECTED pretraining states are saved to EXISTING pretraing states")
                    print("Duplicated states are removed")
                    self.collected_pretraining_states = []

                self.show_states_status()
                return self.existing_pretraining_states
            
            elif comfirm_key_2 == 'n':
                self.show_states_status()
                return self.existing_pretraining_states
            else:
                print("illegal key")
                continue

    def initialize_existing_pretraining_states(self):
        self.show_states_status()
        print("Do you want to initialize the EXISTING pretraining states? (y/n)")
        print("This will remove all the pretraining states inheritated")
        while(True):
            comfirm_key_2 = getch.getch()
            if comfirm_key_2 == 'y':
                np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity), [-1])
                print("OK Proceed with the game.")
                break
            elif comfirm_key_2 == 'n':
                print("OK Proceed with the game.")
                break
            else:
                print("illegal key")
                continue

    def initialize_collected_pretraining_states(self):
        self.show_states_status()
        print("Do you want to initialize the COLLECTED pretraining states? (y/n)")
        print("This will remove all the pretraining states collected in this session")
        while(True):
            comfirm_key_2 = getch.getch()
            if comfirm_key_2 == 'y':
                self.collected_pretraining_states = []
                print("OK Proceed with the game.")
                break
            elif comfirm_key_2 == 'n':
                print("OK Proceed with the game.")
                break
            else:
                print("illegal key")
                continue


    def show_states_status(self):
        self.existing_pretraining_states = np.load("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity))["arr_0"]
        existing_s = self.existing_pretraining_states.shape[0]
        if (len(self.collected_pretraining_states)==0):
            collected_s = 0
        else:
            collected_s = np.unique(self.collected_pretraining_states, axis=0).shape[0]
        print("===========================================================================================")
        if self.existing_pretraining_states.shape == (1,):
            existing_s -= 1
        
        print("Number of EXISTING pretraining states: {}".format(existing_s))
        print("Number of COLLECTED pretraining states: {}".format(collected_s))
        print("Total number of states: {}".format(existing_s+collected_s))
        print("Total number of states needed to at least do clustering (number of clusters): {}".format(self.n_clusters))
        print("Resize factor {}".format(self.s_e_p.resize_factor))
        print("final img size {} * {} = {}".format(self.s_e_p.final_shape[1], self.s_e_p.final_shape[0], self.s_e_p.final_size))
        print("Pixel Intensity: {}".format(self.s_e_p.pixel_intensity))
        print("Pixel Block size: {}".format(self.s_e_p.pixel_block))
        print("Number of steps taken: {}".format(self.steps))
        print("Number of episodes: {}".format(self.current_episode))
        print("Sample Collection Interval: {}".format(self.s_c_i))
        print("===========================================================================================")

        # print(pretraining_states["arr_0"].shape)
        # print(np.array(self.collected_pretraining_states).shape) 

    def print_rule(self):
        print("===========================================================================================")
        print("Pretraining session! Let Mario explore as much as possibile!!")
        print("BASIC MOVES: Arrows. Space is jump forward (right+A+B)")
        print("Otherwise, 0 to {} to play, Press q to quit.".format(self.env.action_space.n-1))
        print("If sufficient number of states are collected, you can quit")
        print("Press p to see Pretraining states status")
        print("Press c to initialize Collected pretraining states")
        print("Press e to initialize Existing pretraining states")
        print("Press s to Save collected pretraining states")
        print("Press r to show this Rule again")
        print("Press t to train randomly")
        print("===========================================================================================")


    def random_walk(self):
        self.cluster = 0
        self.paused = False
        
        while not self.paused:
            if self.episode_done:
                self.next_episode()
                
            self.steps += 1
            
            self.done = False
            accumulated_reward = 0
            
            for frame in range(self.action_interval+3):
                next_state, reward, self.done, info = self.env.step(self.action)
                accumulated_reward += reward
                
                if reward == -15:
                    self.done = True
                if self.done:
                    break
                

            if self.cluster != 0:
                imgArray = self.encode_state(self.clustering_method,
                                             next_state,
                                             self.s_e_p)
                prediction = self.cluster.predict_state_cluster(next_state)
                A = (imgArray//self.s_e_p.pixel_block)
                B = (self.cluster_model.cluster_centers_[prediction]//self.s_e_p.pixel_block)
                internal_reward = np.sum(np.square(A-B))/(self.s_e_p.pixel_intensity**2) -4
                print(accumulated_reward, internal_reward)
                accumulated_reward += internal_reward
                
            if self.steps % self.s_c_i == 0:
                imgArray = self.encode_state(self.clustering_method,
                                             next_state,
                                             self.s_e_p)
                self.collected_pretraining_states.append(imgArray)
            self.env.render()
            
            if self.done:
                self.episode_done = True
                
                self.q_estimator.record_transition(action=self.action,
                                                   reward=accumulated_reward,
                                                   state=next_state.copy(),
                                                   terminal=True,
                                                   lp_action=None)

                self.q_estimator.episode_finished()
                self.action_policy.episode_finished()


                if self.current_episode % 5 == 4:
                    if self.existing_pretraining_states.shape == (1,):
                        self.existing_pretraining_states = np.array(self.collected_pretraining_states)
                    else:
                        self.existing_pretraining_states = np.concatenate([self.existing_pretraining_states, np.array(self.collected_pretraining_states)], 0)

                    self.existing_pretraining_states = np.unique(self.existing_pretraining_states, axis=0)

                    np.savez_compressed("./pretraining_states_ds{}_pi{}.npz".format(self.s_e_p.resize_factor, self.s_e_p.pixel_intensity), self.existing_pretraining_states)


                    print("COLLECTED pretraining states are saved to EXISTING pretraing states")
                    print("Duplicated states are removed")

                    self.collected_pretraining_states = []
                    
                    if self.existing_pretraining_states.shape[0] >= self.n_clusters:
                        C = Cluster(state_encoding_params = self.s_e_p,
                                    action_space_size = self.env.action_space.n,
                                    clustering_method = self.clustering_method,
                                    n_clusters = self.n_clusters)
                        
                        self.cluster_model = C.cluster(self.existing_pretraining_states)
                       
                        self.cluster = C
                
                        C.save_cluster_image()


                        
                        
                        
                    else:
                        print("number of unique samples: too small. resume")
                

            else:
                next_action = self.action_policy.get_action(next_state, self.q_estimator)
                if self.learning_policy == LearningPolicy.SARSA:
                    lp_action = next_action
                elif self.learning_policy == LearningPolicy.Q:
                    lp_action = None

                self.q_estimator.record_transition(action=self.action,
                                                   reward=accumulated_reward,
                                                   state=next_state.copy(),
                                                   terminal=False,
                                                   lp_action=lp_action)


                # We *must* copy state (which is of type ndarray), otherwise, we
                # just get a reference to the mutating state
                self.state = next_state.copy()
                self.action = next_action

        
    
    # Returns pretraining states with encoding
    def get_pretraining_states(self):
        self.print_rule()
        self.initialize_existing_pretraining_states()        
                
        # Control action from keyboard
        while (True):
            if self.done:
                state = self.env.reset()

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
                self.steps += 1
                for frame in range(self.action_interval+3):
                    next_state, reward, self.done, info = self.env.step(action)
                    if reward == -15:
                        self.done = True
                    if self.done:
                        break

                if self.steps % self.s_c_i == 0:
                    imgArray = self.encode_state(self.clustering_method,
                                                 next_state,
                                                 self.s_e_p)
                    self.collected_pretraining_states.append(imgArray)
                    self.env.render()
                
            # Press q to quit
            elif action == 113-48: # q
                while(True):
                    print("Are you sure you want to stop pretraining? (y/n)")
                    comfirm_key = getch.getch()
                    if comfirm_key == 'y':
                        return self.save_pretraining_states()
                    elif comfirm_key == 'n':
                        break
                    else:
                        print("illegal key")
                        continue
                        
            elif action == 112-48: # p
                self.show_states_status()
                
            elif action == 99-48: # c
                self.initialize_collected_pretraining_states()

            elif action == 101-48: # e
                self.initialize_existing_pretraining_states()
                    
            elif action == 114-48: # r
                self.print_rule()

            elif action == 115-48: # s
                self.save_pretraining_states()
                
            elif action == 116-48: # t
                print("Press Ctrl-C to quit random training")
                self.random_walk()
            else:
                continue

        print("End of pretraining session")
        return self.save_pretraining_states()    
        
