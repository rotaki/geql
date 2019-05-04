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
                         fitness):
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
                 learning_policy=LearningPolicy.Q,
                 action_interval = 6,
                 listener=None):
        self.q_estimator = q_estimator
        self.action_policy = action_policy
        self.env = environment
        self.action_list = list(range(self.env.action_space.n))
        self.action_interval = action_interval
        self.learning_policy = learning_policy
        self.listener = listener
        self.current_episode = 0
        self.render_option = RenderOption.ActionFrames
        self.episode_done = True

    def next_episode(self):
        self.current_episode += 1
        self.state = self.env.reset()
        self.action = self.action_policy.get_action(self.state, self.q_estimator)
        self.max_x = 0
        self.time_max_x = 0
        self.time_start = time.monotonic()
        self.frames = 0
        self.episode_done = False
    
    def step(self):
        if self.episode_done:
            self.next_episode()
        done = False
        accumulated_reward = 0
        # Take the pending action for the next n frames
        for frame in range(self.action_interval):
            next_state, reward, done, info = self.env.step(self.action)
            if self.render_option == RenderOption.All:
                self.env.render()

            self.frames += 1
            if info['x_pos'] > self.max_x:
                self.max_x = info['x_pos']
                self.time_max_x = info['time']
    
            accumulated_reward += reward
            # Terminate the episode on death-signal
            if reward == -15:
                done = True
            if done:
                break

        if self.render_option == RenderOption.ActionFrames:
            self.env.render()
            
        if done: # next_state is terminal
            self.episode_done = True
            # If state is terminal, there is no difference between Q and SARSA
            self.q_estimator.reward(self.state,
                                    self.action,
                                    accumulated_reward,
                                    next_state,
                                    None)

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
                    self.max_x)
            return False
        else: # next_state is *not* terminal
            next_action = self.action_policy.get_action(next_state, self.q_estimator)
            if self.learning_policy == LearningPolicy.SARSA:
                q_update_action = next_action
            elif self.learning_policy == LearningPolicy.Q:
                # Select action with highest Q(next_state, next_action)
                action_values = self.q_estimator.batch_estimate(next_state,
                                                            self.action_list)
                av_pair = max(action_values, key=lambda av: av[0])
                q_update_action = av_pair[0]
                self.q_estimator.reward(self.state,
                                        self.action,
                                        accumulated_reward,
                                        next_state,
                                        q_update_action)
            self.state = next_state
            self.action = next_action

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
    
    env = gym_smb.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
    action_list = list(range(env.action_space.n))
    action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.1)
    learning_policy = LearningPolicy.SARSA
    q_estimator = TabQ.TabularQEstimator(discount=0.5, learning_rate=0.2)
    listener = PlotOnlyListener(q_estimator, action_policy, learning_policy)
    agent = MarioRLAgent(env, q_estimator, action_policy, learning_policy, listener=listener)
    for i in range(10000):
        agent.step()
    env.close()
