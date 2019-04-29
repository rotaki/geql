from RLInterfaces import IActionPolicy
import random

class EpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon):
        self.epsilon = epsilon
        self.actions = actions

    def action(self, state, q_estimator):
        if random.random() < self.epsilon:
            # Choose randomly
            return random.choice(self.actions)
        else:
            # Choose greedily
            return max(self.actions, key=lambda a:q_estimator.estimate(state, a))

    # TODO: Load/save
