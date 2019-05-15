from RLInterfaces import IActionPolicy
import numpy as np
import random

class ClusterEpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon):
        self.epsilon = epsilon
        self.actions = actions
        self.c = None
        self.action_counter = None

    def initialize_cluster_model(self, cluster_model):
        self.c = cluster_model
        self.action_counter = np.zeros((self.c.n_clusters, len(self.actions)))
        

    def gibbs_action_count(self, cluster):
        temp = np.exp(-1*self.action_counter[cluster])
        temp = temp/np.sum(temp)
        return temp

    def show_action_count(self):
        if self.action_counter is not None:
            return self.action_counter
        else:
            return

    # Returns action count of a cluster
    def action_count(self, cluster):
        return self.action_counter[cluster]

        # Add one to the count when action taken from a state
    def add_action_count(self, cluster, action):
        self.action_counter[cluster][action] += 1

        
    def get_action(self, state, q_estimator):
        if random.random() < self.epsilon:
            # Choose randomly
            if self.c is not None:
                mask = self.gibbs_action_count(self.c.predict_state_cluster(state))
                action_choice = np.random.choice(self.actions, p=mask)
                self.add_action_count(self.c.predict_state_cluster(state), action_choice)
            else:
                action_choice = np.random.choice(self.actions)
            return action_choice
        
        else:
            # Choose greedily (break ties randomly)
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_v = max(action_values, key=lambda av: av[1])[1]
            candidates = list(filter(lambda av: av[1] == best_v, action_values))
            chosen = random.choice(candidates)
            if self.c is not None:
                self.add_action_count(self.c.predict_state_cluster(state), chosen[0])
            return chosen[0]

    def summary(self):
        return 'Cluster $\epsilon-greedy$ [$\epsilon = {}$, $|A| = {}$]'.format(self.epsilon, len(self.actions))

    # TODO: Load/save
