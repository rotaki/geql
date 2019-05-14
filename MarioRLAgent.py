from enum import Enum
import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros.actions import *
import time

import TrainingStats as TS
import impl.EpsilonGreedyActionPolicy as EGAP
import impl.TabularQEstimator as TabQ

class LearningPolicy(Enum):
    Q = 0
    SARSA = 1
    def describe(p):
        if p is LearningPolicy.Q:
            return 'Q-Learning'
        if p is LearningPolicy.SARSA:
            return 'SARSA'
        return 'Unknown'

class RenderOption(Enum):
    NoRender = 0 
    ActionFrames = 1
    All = 2
    
class IMarioRLAgentListener:
    """
    Listener interface for a MarioRLAgent
    """
    
    def episode_finished(self,
                         episode_number,
                         wall_time_elapsed,
                         game_time_elapsed,
                         n_frames,
                         fitness,
                         sa_sequence
    ):
        """
        Called by MarioRLAgent every time an episode finishes

        Parameters
        ----------
        episode_number : int
            Monotonically increasing sequence number of the finished episode
        wall_time_elapsed : float
            CPU time (in seconds) taken to complete the episode, including time 
            spent postprocessing q_estimator and action_policy
        game_time_elapsed : int
            Game time (in game seconds) taken to complete the episode
        n_frames : int
            Number of frames in episode
        fitness : numeric
            Fitness value as defined by the MarioRLAgent implementation, where
            higher is better
        sa_sequence : (state, action)
            List of the entire episode as state, action-pairs

        Returns
        -------
        Nothing
        """
        raise NotImplementedError()
    
class MarioRLAgent:
    """
    Class for letting a RL agent train on, and play Mario. This could probably
    be refactored into a general OpenAI Gym RL agent, but for now, there's some
    specialization for the SuperMarioBros environment here
    """
    def __init__(self,
                 environment,
                 q_estimator,
                 action_policy,
                 action_set,
                 learning_policy=LearningPolicy.Q,
                 action_interval = 6,
                 listener=None):
        self.env = environment
        self.q_estimator = q_estimator
        self.action_policy = action_policy
        self.action_set = action_set
        self.action_list = list(range(self.env.action_space.n))
        self.action_interval = action_interval
        self.learning_policy = learning_policy
        self.listener = listener
        self.current_episode = 0
        self.render_option = RenderOption.ActionFrames
        self.game_over = True
        self.episode_done = True
        self.verbose = False
        self.hsep = '================================================================================'
        self.kill_timer = 10        
        self.sa_sequence = []

    def next_episode(self):
        self.time_start = time.monotonic()
        self.current_episode += 1
        if self.verbose:
            print('Starting episode {}'.format(self.current_episode))
        if self.game_over:
            self.state = self.env.reset()
            self.game_over = False
        self.q_estimator.episode_start(self.state.copy())
        self.action = self.action_policy.get_action(self.state, self.q_estimator)
        self.max_x = 0
        self.time_max_x = 0
        self.frames = 0
        self.sa_sequence = []
        self.episode_done = False

    def best_action(self, state):
        """
        Returns
        -------
        (action, float)
            Tuple containing the index of the best action, together with its q-value
        """
        action_values = self.q_estimator.batch_estimate(state, self.action_list)
        return max(action_values, key=lambda av: av[1])
            
    def format_all_q_values(self, state, selected_action):
        best = self.best_action(state)
        action_values = self.q_estimator.batch_estimate(state, self.action_list)
        result_str = '\t {:2} {:20} {}\n'.format('id', 'Action', 'Q(s, a)')
        for (i, (a, v)) in enumerate(action_values):
            append_best = ' (best)' if v == best[1] else ''
            append_selected = ' (selected)' if a == selected_action else ''
            result_str = result_str + '\t {:2} {:20} {} {}{} \n'.format(
                i, str(self.action_set[a]), v, append_best, append_selected)
        return result_str
    
    def step(self):
        if self.episode_done:
            self.next_episode()

        if self.verbose:
            print(self.hsep)

        accumulated_reward = 0
        
        # Take the pending action for the next n frames
        for frame in range(self.action_interval):
            self.sa_sequence.append((self.state, self.action))
            next_state, reward, self.game_over, info = self.env.step(self.action)
            self.episode_done = self.game_over
                        
            if info['x_pos'] > 60000:
                print('Warning: Ignoring insane x_pos {}'.format(info['x_pos']))
            elif info['x_pos'] > self.max_x:
                self.max_x = info['x_pos']
                self.time_max_x = info['time']
            elif info['time'] + self.kill_timer < self.time_max_x:
                # Kill mario if standing still too long
                reward = -15
                self.game_over = True
                self.episode_done = True

            # Terminate the episode on death-signal
            if reward <= -15:
                self.episode_done = True

            # Teriminate the game on finishing the level
            if info['flag_get']:
                self.game_over = True
                self.episode_done = True
                
            accumulated_reward += reward

            if self.render_option == RenderOption.All:
                self.env.render()
            
            if self.verbose:
                is_action_frame = frame == self.action_interval - 1
                print('\nFrame: {} Action frame: {}'.
                      format(self.frames, is_action_frame))
                print('\t {:14} {}'.format('reward', reward))
                print('\t {:14} {}'.format('acc. reward', accumulated_reward))
                print('\t {:14} {}'.format('episode done', self.episode_done))
                print('\t {:14} {}'.format('game over', self.game_over))
                for key, value in info.items():
                    print('\t {:14} {}'.format(key, value))
                print('\t {:14} {}'.format('max x', self.max_x))
                print('\t {:14} {}'.format('time max x', self.time_max_x))
                
            self.frames += 1
    
            if self.episode_done:
                break
        
        if self.render_option == RenderOption.ActionFrames:
            self.env.render()
            
        if self.episode_done: # next_state is terminal
            self.q_estimator.record_transition(action=self.action,
                                               reward=accumulated_reward,
                                               state=next_state.copy(),
                                               terminal=True,
                                               lp_action=None)
            
            self.q_estimator.episode_finished()
            self.action_policy.episode_finished()
            # Record fitness variables
            # Important: stop timer *after* batch-updates for fair FPS-comparison
            time_elapsed = time.monotonic() - self.time_start
            # Listener
            if self.listener is not None:
                self.listener.episode_finished(
                    self.current_episode,
                    time_elapsed,
                    400 - self.time_max_x,
                    self.frames,
                    self.max_x,
                    self.sa_sequence
                )

        else: # next_state is *not* terminal
            next_action = self.action_policy.get_action(next_state, self.q_estimator)
            
            if self.verbose:
                print('\n' + self.format_all_q_values(next_state, next_action))
            
            if self.learning_policy == LearningPolicy.SARSA:
                lp_action = next_action
            elif self.learning_policy == LearningPolicy.Q:
                lp_action = None

            self.q_estimator.record_transition(action=self.action,
                                               reward=accumulated_reward,
                                               state=next_state.copy(),
                                               terminal=False,
                                               lp_action=lp_action)

            # We *must* copy state (which is of type ndarray), otherwise, we
            # just get a reference to the mutating state
            self.state = next_state.copy()
            self.action = next_action

        if self.verbose:
            print(self.hsep)


if __name__ == '__main__':
    print('Starting MarioRLAgent *without* UI. This is a debugging mode and' +
          ' probably not what you want unless you know what you\'re doing')

    class PlotOnlyListener(IMarioRLAgentListener):
        def __init__(self, q_estimator, action_policy, learning_policy):
            self.training_stats = \
                TS.TrainingStats(q_estimator,
                                 action_policy,
                                 'Learning policy: {}'.
                                 format(LearningPolicy.describe(learning_policy)))
            self.training_stats.plot()

        def episode_finished(self,
                             episode_number,
                             wall_time_elapsed,
                             game_time_elapsed,
                             n_frames,
                             fitness):
            self.training_stats.add_episode_stats(
                wall_time_elapsed,
                game_time_elapsed,
                n_frames,
                fitness)
            self.training_stats.plot()

    action_set = RIGHT_ONLY
    env = gym_smb.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, action_set)
    action_list = list(range(env.action_space.n))
    action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.1)
    learning_policy = LearningPolicy.SARSA
    q_estimator = TabQ.TabularQEstimator(discount=0.5, learning_rate=0.2)
    listener = PlotOnlyListener(q_estimator, action_policy, learning_policy)
    agent = MarioRLAgent(env, q_estimator, action_policy, action_set, learning_policy, listener=listener)
    for i in range(10000):
        agent.step()
    env.close()
