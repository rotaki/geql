from RLInterfaces import IActionPolicy
import random

class EpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon):
        self.epsilon = epsilon
        self.actions = actions

    def get_action(self, state, q_estimator):
        if random.random() < self.epsilon:
            # Choose randomly
            return random.choice(self.actions)
        else:
            # Choose greedily
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_av = max(action_values, key=lambda av: av[1]) 
            return best_av[0]

    def summary(self):
        return '$\epsilon-greedy$ [$\epsilon = {}$, $|A| = {}$]'.format(self.epsilon, len(self.actions))

    # TODO: Load/save
