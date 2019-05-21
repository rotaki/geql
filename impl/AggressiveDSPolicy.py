from RLInterfaces import IActionPolicy
from EncodeState import EncodeState
import numpy as np
import random

class AggressiveDSPolicy(IActionPolicy):
    def __init__(self, actions, epsilon, state_encoding_params, decay_factor=1, decay_interval=10000):
        self.epsilon = epsilon
        self.actions = actions
        self.s_e_p = state_encoding_params
        self.action_counter = dict()
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.episodes_seen = 0
        self.current_epsilon = epsilon
        
    def show_action_count(self):
        return self.action_counter

    def encode_state(self, state):
        return EncodeState().encode_state(clustering_method="agressive_ds",
                                          state=state,
                                          state_encoding_params=self.s_e_p)

    def add_action_count(self, encoded_state, action):
        self.action_counter[encoded_state][action] += 1
        
    def gibbs_action_count(self, encoded_state):
        temp = np.nan_to_num(np.exp(-1* np.array(self.action_counter[encoded_state])))
        temp = np.nan_to_num(temp/np.sum(temp))
        return temp

    def get_action(self, state, q_estimator):
        encoded_state = self.encode_state(state)
        if encoded_state not in self.action_counter:
            self.action_counter[encoded_state] = [0 for i in range(len(self.actions))]
        if random.random() < self.current_epsilon:
            mask = self.gibbs_action_count(encoded_state)
            try:
                action_choice = np.random.choice(self.actions, p=mask)
            except:
                action_choice = np.random.choice(self.actions)
            self.add_action_count(encoded_state, action_choice)
            return action_choice
        else:
            # Choose greedily (break ties randomly)
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_v = max(action_values, key=lambda av: av[1])[1]
            candidates = list(filter(lambda av: av[1] == best_v, action_values))
            chosen = random.choice(candidates)
            self.add_action_count(encoded_state, chosen[0])
            return chosen[0]

    def episode_finished(self):
        self.episodes_seen += 1
        self.current_epsilon = self.epsilon * pow(self.decay_factor, self.episodes_seen / self.decay_interval)
        print('Current epsilon: {}'.format(self.current_epsilon))

    def summary(self):
        return 'Agressive DS $\epsilon-greedy$ [$\epsilon = {}$, $|A| = {}$, $downscale factor = {}$, $pixel intensity = {}$]'.format(self.epsilon, len(self.actions), self.s_e_p.resize_factor, self.s_e_p.pixel_intensity)
    
