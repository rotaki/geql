from RLInterfaces import IQEstimator
import numpy as np

class TabularQEstimator (IQEstimator):
    def __init__(self, actions, discount, learning_rate, policy=None):
        self.q_table = dict()
        self.actions = actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.policy = policy

    def transform_state(state):
        # Skip the n top rows to get rid of timing/blinking coin to shrink
        # the state space a bit
        return state[49:,0:,0:]

    def estimate(self, state, action):
        # If the state-action tuple isn't in the table, return 0
        return self.q_table.get((transform_state(state).tobytes(), action), 0.0) 
        
    def reward(self, state, action, reward, result_state):
        old_estimate = self.estimate(state, action)
        if self.policy is None:
            # Q-learning
            result_state_value = max([self.estimate(result_state, a)
                                      for a in self.actions])
        else:
            # SARSA
            result_state_value = self.estimate(result_state,
                                               self.policy.action(result_state, self))
        temporal_error = reward + self.discount * result_state_value - old_estimate
        new_estimate = old_estimate + self.learning_rate * temporal_error
        self.q_table[(transform_state(state).tobytes(), action)] = new_estimate

    def batch_reward(self, sars_list):
        for sars in sars_list:
            self.reward(sars)

    # TODO: save/load
