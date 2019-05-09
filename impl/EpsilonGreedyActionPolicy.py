from RLInterfaces import IActionPolicy
import numpy as np
import random

class EpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon, cluster):
        self.epsilon = epsilon
        self.actions = actions
        self.cluster = cluster if cluster is not None else None
        
    def get_action(self, state, q_estimator):
        if random.random() < self.epsilon:
            # Choose randomly
            if self.cluster is not None:
                mask = self.cluster.gibbs_action_count(state)
                action_choice = np.random.choice(self.actions, p=mask)
                self.cluster.add_action_count(state, action_choice)
            else:
                action_choice = np.random.choice(self.actions)
            return action_choice
        
        else:
            # Choose greedily (break ties randomly)
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_v = max(action_values, key=lambda av: av[1])[1]
            candidates = list(filter(lambda av: av[1] == best_v, action_values))
            chosen = random.choice(candidates)
            if self.cluster is not None:
                self.cluster.add_action_count(state, chosen[0])
            return chosen[0]

    def summary(self):
        return '$\epsilon-greedy$ [$\epsilon = {}$, $|A| = {}$]'.format(self.epsilon, len(self.actions))

    # TODO: Load/save
