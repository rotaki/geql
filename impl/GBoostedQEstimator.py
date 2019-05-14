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

    def estimate(self, dmatrix):
        accumulator = 0.0
        for (alpha, regressor) in self.regressors:
            accumulator += alpha * regressor.predict(dmatrix,
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

        self.trajectories = []
        self.n_trajectories = 0
        self.initial_trajectories_per_regressor = 64
        self.trajectories_per_regressor = self.initial_trajectories_per_regressor
        self.max_trajectories_per_regressor = 1024
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
        state_dmatrix = xgboost.DMatrix(self.shape_state(state))
        return self.estimate_dmatrix(state_dmatrix, action)

    def estimate_dmatrix(self, state_dmatrix, action):
        if action not in self.estimators:
            return 0.0
        else:
            return self.estimators[action].estimate(state_dmatrix)

    def batch_estimate(self, state, actions):
        state_dmatrix = xgboost.DMatrix(self.shape_state(state))
        return list(map(
            lambda a: (a, self.estimate_dmatrix(state_dmatrix, a)),
            actions))

    def episode_start(self, initial_state):
        self.trajectories.append(Trajectory(initial_state))
        self.n_trajectories += 1
        self.sa_tuple_count = 0
        if self.verbose:
            print('Initial state s=<{:x}>'.format(hash(initial_state.tobytes()) % self.hashmod))

    def record_transition(self, action, reward, state, terminal, lp_action):
        self.trajectories[self.n_trajectories - 1].add_transition(action,
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
        # Do nothing unless we have the trajectories
        if self.n_trajectories < self.trajectories_per_regressor:
            print('{}/{} trajectories finished for next regressor'.format(self.n_trajectories,
                                                                              self.trajectories_per_regressor))
            return
        print('{}/{} training regressor...'.format(self.n_trajectories, self.trajectories_per_regressor))
        residuals_by_action = dict()

        # Process the trajectories one by one
        for trajectory in self.trajectories:
            # Get Q(s, a) observations for this episode
            if self.learning_policy == MarioRLAgent.LearningPolicy.Q:
                sa_tuples = trajectory.q_episode_backup(self.discount,
                                                             self.steps,
                                                             self,
                                                             self.q_action_policy)
            elif self.learning_policy == MarioRLAgent.LearningPolicy.SARSA:
                sa_tuples = trajectory.sarsa_episode_backup(self.discount,
                                                                 self.steps,
                                                                 self)
            else:
                raise NotImplementedError('Unknown learning policy')

            # State size
            state_size = self.shape_state(sa_tuples[0].state).size        
            # Calculate residuals grouped by action

            if self.verbose:
                print('Calculating residuals by action')
            for sa_tuple in sa_tuples:
                if sa_tuple.action not in residuals_by_action:
                    residuals_by_action[sa_tuple.action] = list()
                residuals_by_action[sa_tuple.action].append(
                    (self.shape_state(sa_tuple.state),
                     self.estimate(sa_tuple.state, sa_tuple.action) - sa_tuple.q)
                )

        for (action, residuals) in residuals_by_action.items():
            n = len(residuals)
            # Form the data matrix that will be used to train this regressor
            # (There is probably a more efficient way to do all this data juggling)
            state_matrix = np.empty([n, state_size])
            target_vector = np.empty([n, 1])
            residual_squared = 0
            for i, residual in enumerate(residuals):
                state_matrix[i] = residual[0]
                target_vector[i] = -residual[1] # We want to train h to cancel out the residual
                residual_squared += residual[1] * residual[1]
            
            residual_squared /= n
            residual_squared = np.sqrt(residual_squared)
            print('Prior rmse for action {}: {}'.format(action, residual_squared))
            dmatrix = xgboost.DMatrix(state_matrix, target_vector)
            if self.trajectories_per_regressor == self.max_trajectories_per_regressor:
                params = dict([
                    ('max_depth', 5),
                    #('gpu_id', 0),
                    #('tree_method', 'gpu_exact'),
                  
                ])
            else:
                params = dict([
                    ('max_depth', 3)
                    #('gpu_id', 0),
                    #('tree_method', 'gpu_exact')
                ])
            evallist = [(dmatrix, 'action-{}'.format(action))]
            print('Training regressor for action {}. n={}'.format(action, n))
            booster = xgboost.train(params, dmatrix, evals=evallist, verbose_eval=True)

            # Add the booster to the set of regressors for the action
            if action not in self.estimators:
                self.estimators[action] = ActionEstimator()
            self.estimators[action].add_regressor(self.learning_rate, booster.copy())

        # Discard the trajectories that were used
        self.n_trajectories = 0
        self.trajectories = []
        self.trajectories_per_regressor = min(self.max_trajectories_per_regressor,
                                              2 * self.trajectories_per_regressor)

    def shape_state(self, state):
        """
        Returns a copy of state suitable for storing over longer time
        (downsampled to "final" size, compressed etc
        """
        i = Image.fromarray(state)

        # Conver to grayscale
        # i = i.convert(mode='L')

        # Downsample
        width, height = i.size
        i = i.resize((round(width / self.downsampling), round(height / self.downsampling)),
                     resample=Image.NEAREST)

        # Flatten into 1d array
        return np.array(i).reshape(1, -1)



