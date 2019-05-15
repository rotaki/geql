from RLInterfaces import IQEstimator
from Trajectory import *
import MarioRLAgent
import numpy as np
import zlib
import pickle
from PIL import Image

import time

class TabularQEstimator (IQEstimator):
    def __init__(self,
                 discount,
                 steps,
                 learning_rate,
                 learning_policy,
                 q_action_policy):

        self.q_table = dict()
        self.discount = discount
        self.steps = steps
        self.learning_rate = learning_rate
        self.q_action_policy = q_action_policy
        self.learning_policy = learning_policy
        
        if self.learning_policy == MarioRLAgent.LearningPolicy.Q:
            if self.steps != 1:
                raise NotImplementedError('Q is only supported with steps=1')
            if self.q_action_policy is None:
                raise NotImplementedError('Q requires a q_action_policy')
        
        self.compression = 6
        self.downsampling = 8
        self.hits = 0
        self.misses = 0

        self.hashmod = pow(16,4)
        
        self.sa_tuple_count = 0

        self.verbose = False

    def summary(self):
        return 'TabQ [$\\alpha={}$, $\\gamma={}$, $n={}$]'.format(self.learning_rate,
                                                                  self.discount,
                                                                  self.steps)

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
    
    def episode_start(self, initial_state):
        self.trajectory = Trajectory(initial_state)
        self.sa_tuple_count = 0
        if self.verbose:
            print('Initial state s=<{:x}>'.format(hash(initial_state.tobytes()) % self.hashmod))

    def record_transition(self, action, reward, state, terminal, lp_action):
        self.trajectory.add_transition(action,
                                       reward,
                                       state,
                                       terminal,
                                       lp_action)
        if self.verbose:
            print('Record #{}: a={}, r={}, s\'=<{:x}>{}, a\'={}'.format(
                self.sa_tuple_count,
                action,
                reward,
                hash(state.tobytes()) % self.hashmod,
                ' (terminal)' if terminal else '',
                lp_action
            ))
            
        self.sa_tuple_count += 1
        if self.sa_tuple_count >= self.steps:
            self.update(self.sa_tuple_count - self.steps)

        if terminal:
            # Also update the remaining steps (if any)
            if self.verbose:
                print('Updating remaining steps at end of episode')
            for t in range(self.steps - 1, 0, -1):
                self.update(self.sa_tuple_count - t)

    def update(self, sa_tuple_entry):
        if self.learning_policy == MarioRLAgent.LearningPolicy.Q:
            sa_tuple = self.trajectory.q_backup(sa_tuple_entry,
                                                self.discount,
                                                self.steps,
                                                self,
                                                self.q_action_policy)
        elif self.learning_policy == MarioRLAgent.LearningPolicy.SARSA:
            sa_tuple = self.trajectory.sarsa_backup(sa_tuple_entry,
                                                    self.discount,
                                                    self.steps,
                                                    self)
        else:
            raise NotImplementedError('Unknown learning policy')

        Q_sa = self.estimate(sa_tuple.state, sa_tuple.action)
        Q_sa_observed = sa_tuple.q
        temporal_error = Q_sa_observed - Q_sa
        # This is based on the "naive" n-step off policy (7.3 in Sutton)
        # "Per-decision Methods with Control Variates" (7.4) may be more effective
        Q_sa_updated = Q_sa + self.learning_rate * sa_tuple.importance * temporal_error

        if self.verbose:
            print('Update #{} Q(<{:x}>, {}): prior={}, observation={}, importance={}, posterior={}'.
                  format(sa_tuple_entry,
                         hash(sa_tuple.state.tobytes()) % self.hashmod,
                         sa_tuple.action,
                         Q_sa,
                         Q_sa_observed,
                         sa_tuple.importance,
                         Q_sa_updated))
        
        self.q_table[(self.encode_state(sa_tuple.state), sa_tuple.action)] = Q_sa_updated

    def episode_finished(self):
        # Print some stats
        queries = self.hits + self.misses
        print('TabQ hit ratio: {:.2f} % ({} hits, {} misses)'.
              format(self.hits/queries * 100, self.hits, self.misses))
        self.hits = 0
        self.misses = 0
    
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

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def from_file(filename):
        return pickle.load(open(filename, 'rb'))
