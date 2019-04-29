class IQEstimator:
    def estimate(self, state, action):
        raise NotImplementedError('estimate not implemented')

    def reward(self, state, action, reward, result_state):
        raise NotImplementedError('reward not implemented')

    def batch_reward(self, sars_list):
        raise NotImplementedError('batch_reward not implemented')

    def save(self, filename):
        raise NotImplementedError('save not implemented')

    def from_file(filename):
        raise NotImplementedError('from_file not implemented')

class IActionPolicy:
    def action(self, state, q_estimator):
        raise NotImplementedError('action not implemented')

    def save(self, filename):
        raise NotImplementedError('save not implemented')

    def from_file(filename):
        raise NotImplementedError('from_file not implemented')

class ITerminationPolicy:
    def new_episode(self):
        raise NotImplementedError('new_episode not implemented')

    def update_and_decide(self):
        raise NotImplementedError('update_and_decide not implemented')
