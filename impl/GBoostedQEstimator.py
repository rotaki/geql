from RLInterfaces import IQEstimator
from Trajectory import *
from collections import namedtuple
from PIL import Image
import numpy as np
import xgboost
import MarioRLAgent

class ActionEstimator:
    def __init__(self):
        self.regressors = []

    def estimate(self, shaped_state):
        accumulator = 0.0
        shape2 = shaped_state.reshape([1, shaped_state.size])
        data = xgboost.DMatrix(shape2)
        for (alpha, regressor) in self.regressors:
            accumulator += alpha * regressor.predict(data,
                                                     validate_features=False)
        return float(accumulator)

    def add_regressor(self, alpha, regressor):
        self.regressors.append((alpha, regressor))

class GBoostedQEstimator(IQEstimator):
    def __init__(self, discount,
                 steps,
                 learning_rate,
                 learning_policy,
                 q_action_policy):
        
        self.trajectory = []
        self.estimators = dict()
        self.discount = discount
        self.learning_rate = learning_rate
        self.steps = steps
        if steps > 1 and learning_policy != MarioRLAgent.LearningPolicy.SARSA:
            # Check Sutton chapter 7.3 and 7.5 for possible implementation
            raise NotImplementedError('Not yet implemented')
        self.learning_policy = learning_policy
        self.q_action_policy = q_action_policy
        
        self.hashmod = pow(16,4)
        self.verbose = False

        self.downsampling = 8

        
    def summary(self):
        return 'GBoostQ [$\\alpha={}$, $\\gamma={}$, $n={}$]'.format(
            self.learning_rate,
            self.discount,
            self.steps)
    
    def estimate(self, state, action):
        if action not in self.estimators:
            return 0.0
        else:
            return self.estimators[action].estimate(self.shape_state(state))

#    def estimate_preprocessed_state(self, state, action):
#        return 0.0

    def batch_estimate(self, state, actions):
        # TODO: Reshaping
        return list(map(lambda a: (a, self.estimate(state, a)), actions))

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

        # Only update on terminal (in episode_finished())
    
        
    def episode_finished(self):
        # Get Q(s, a) observations for this episode
        if self.learning_policy == MarioRLAgent.LearningPolicy.Q:
            sa_tuples = self.trajectory.q_episode_backup(self.discount,
                                                         self.steps,
                                                         self,
                                                         self.q_action_policy)
        elif self.learning_policy == MarioRLAgent.LearningPolicy.SARSA:
            sa_tuples = self.trajectory.sarsa_episode_backup(self.discount,
                                                             self.steps,
                                                             self)
        else:
            raise NotImplementedError('Unknown learning policy')

        # State size
        state_size = self.shape_state(sa_tuples[0].state).size        
        # Calculate residuals grouped by action
        residuals_by_action = dict()
        if self.verbose:
            print('Calculating residuals by action')
        for sa_tuple in sa_tuples:
            residual = self.estimate(sa_tuple.state, sa_tuple.action) - sa_tuple.q
            if sa_tuple.action not in residuals_by_action:
                residuals_by_action[sa_tuple.action] = list()
            residuals_by_action[sa_tuple.action].append((sa_tuple.state,
                                          self.estimate(sa_tuple.state, sa_tuple.action) - sa_tuple.q))

        for (action, residuals) in residuals_by_action.items():
            n = len(residuals)
            # Form the data matrix that will be used to train this regressor
            # (There is probably a more efficient way to do all this data juggling)
            state_matrix = np.empty([n, state_size])
            target_vector = np.empty([n, 1])
            residual_squared = 0
            for i, residual in enumerate(residuals):
                state_matrix[i] = self.shape_state(residual[0])
                target_vector[i] = -residual[1] # We want to train h to cancel out the residual
                if self.verbose:
                    residual_squared += residual[1] * residual[1]
            if self.verbose:
                residual_squared /= n
                residual_squared = np.sqrt(residual_squared)
                print('Prior rmse for action {}: {}'.format(action, residual_squared))
            dmatrix = xgboost.DMatrix(state_matrix, target_vector)
            params = dict([
                ('max_depth', 2)
#                ('gpu_id', 0),
#                ('max_bin', 16),
#                ('tree_method', 'gpu_hist'),
#                ('verbosity', 1)
            ])
            evallist = [(dmatrix, 'action-{}'.format(action))]
            if self.verbose:
                print('Training regressor for action {}. n={}'.format(action, n))
            booster = xgboost.train(params, dmatrix, evals=evallist, verbose_eval=self.verbose)

            # Add the booster to the set of regressors for the action
            if action not in self.estimators:
                self.estimators[action] = ActionEstimator()
            self.estimators[action].add_regressor(self.learning_rate, booster.copy())

    def shape_state(self, state):
        """
        Returns a copy of state suitable for storing over longer time
        (downsampled to "final" size, compressed etc
        """
        # TODO: Compression/downsampling ?
        i = Image.fromarray(state)

        # Conver to grayscale
        i = i.convert(mode='L')

        # Downsample
        width, height = i.size
        i = i.resize((round(width / self.downsampling), round(height / self.downsampling)),
                     resample=Image.NEAREST)

        # Flatten into 1d array
        #return state.copy().reshape(-1)
        return np.array(i).reshape(-1)



