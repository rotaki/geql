from RLInterfaces import IActionPolicy
import numpy as np
import random

class ClusterEpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon, cluster):
        self.epsilon = epsilon
        self.actions = actions
        self.cluster = cluster
        
    def get_action(self, state, q_estimator):
        if random.random() < self.epsilon:
            # Choose randomly
            mask = self.cluster.gibbs_action_count(state)
            action_choice = np.random.choice(self.actions, p=mask)
            self.cluster.add_action_count(state, action_choice)
            return action_choice
        
        else:
            # Choose greedily (break ties randomly)
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_v = max(action_values, key=lambda av: av[1])[1]
            candidates = list(filter(lambda av: av[1] == best_v, action_values))
            chosen = random.choice(candidates)
            self.cluster.add_action_count(state, chosen[0])
            return chosen[0]

    def summary(self):
        return 'Cluster $\epsilon-greedy$ [$\epsilon = {}$, $|A| = {}$]'.format(self.epsilon, len(self.actions))

    # TODO: Load/save
