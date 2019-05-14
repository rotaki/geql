from RLInterfaces import IActionPolicy
import random

class EpsilonGreedyActionPolicy (IActionPolicy):
    def __init__(self, actions, epsilon, decay_factor = 1, decay_interval = 10000):
        self.epsilon = epsilon
        self.actions = actions
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.episodes_seen = 0
        self.current_epsilon = epsilon

    def get_action(self, state, q_estimator):
        if random.random() < self.current_epsilon:
            # Choose randomly
            return random.choice(self.actions)
        else:
            # Choose greedily (break ties randomly)
            action_values = q_estimator.batch_estimate(state, self.actions)
            best_v = max(action_values, key=lambda av: av[1])[1]
            candidates = list(filter(lambda av: av[1] == best_v, action_values))
            chosen = random.choice(candidates)
            return chosen[0]

    def episode_finished(self):
        self.episodes_seen += 1
        self.current_epsilon = self.epsilon * \
                               pow(self.decay_factor, self.episodes_seen / self.decay_interval)
        print('Current epsilon: {}'.format(self.current_epsilon))
                
    def summary(self):
        return '$\epsilon-greedy$ [$\epsilon = {}$, decay = ${}^\\frac{{e}}{{{}}}$, $|A| = {}$]'.format(
            self.epsilon,
            self.decay_factor,
            self.decay_interval,
            len(self.actions))

    # TODO: Load/save
