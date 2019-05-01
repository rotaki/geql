from RLInterfaces import IQEstimator
import numpy as np
import zlib

class TabularQEstimator (IQEstimator):
    def __init__(self, actions, discount, learning_rate, policy=None):
        self.q_table = dict()
        self.actions = actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.policy = policy
        self.compression = 6

    def summary(self):
        temporal_mode = 'Q-learning' if self.policy is None else 'SARSA'
        return 'TabQ [{}, $\\alpha={}$, $\\gamma={}$, $|A| = {}$]'.format(temporal_mode,
                                                                          self.learning_rate,
                                                                          self.discount,
                                                                          len(self.actions))
        
    def encode_state(self, state):
        # Skip the n top rows to get rid of timing/blinking coin to shrink
        # the state space a bit
        roi = state[49:,0:,0:]
        # Compression ratio is about 0.008. Worth it!
        return zlib.compress(roi.tostring(), self.compression)

    def estimate(self, state, action):
        # If the state-action tuple isn't in the table, return 0
        return self.q_table.get((self.encode_state(state), action), 0.0) 

    def batch_estimate(self, state, actions):
        state_repr = self.encode_state(state)
        return map(lambda a: (a, self.q_table.get((state_repr, a), 0.0)), actions)
    
    def reward(self, state, action, reward, result_state):
        old_estimate = self.estimate(state, action)
        if self.policy is None:
            # Q-learning
            result_state_value = max(map(lambda av: av[1],
                                         self.batch_estimate(result_state, self.actions)))
        else:
            # SARSA
            result_state_value = self.estimate(result_state,
                                               self.policy.action(result_state, self))
        temporal_error = reward + self.discount * result_state_value - old_estimate
        new_estimate = old_estimate + self.learning_rate * temporal_error
        self.q_table[(self.encode_state(state), action)] = new_estimate

    def episode_finished(self):
        # Do nothing here for this QEstimator
        # (if we wanted to do per-episode updates, we'd do it here)
        pass

    # TODO: save/load
