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
        self.action_list = list(range(env.action_space.n))
        self.action_interval = action_interval
        self.learning_policy = learning_policy
        self.listener = listener
        self.current_episode = 0
        self.keep_running = False

    def run(self):
        if self.keep_running:
            raise RuntimeError('Already running')
        self.keep_running = True

        # Update the plot
        while self.keep_running:
            self.current_episode += 1
            episode_done = False
            state = env.reset()
            max_x = 0
            time_max_x = 0
            time_start = time.monotonic()
            frames = 0
            action = self.action_policy.get_action(state, q_estimator)
            accumulated_reward = 0
            while not episode_done:
                # Take the action
                result_state, reward, episode_done, info = self.env.step(action)

                # Terminate the episode on death-signal
                if reward == -15:
                    episode_done = True

                # Accumulate reward between non-action frames
                accumulated_reward += reward
                self.env.render()
                if frames > 0 and frames % self.action_interval == 0:
                    # These are the only frames the RL-agent "sees"

                    # Action for next state (unless state is terminal)
                    result_state_action = self.action_policy.get_action(
                        result_state, q_estimator) if not episode_done else None

                    # Update q-estimator depends on which learning policy is used
                    if self.learning_policy is LearningPolicy.Q:
                        # Select action2 with highest Q(s, action2)
                        if not episode_done:
                            action_values = self.q_estimator.batch_estimate(
                                result_state, self.action_list)
                            av_pair = max(action_values, key=lambda av: av[0])
                            action2 = av_pair[0]
                        else:
                            action2 = None
                        self.q_estimator.reward(state,
                                                action,
                                                accumulated_reward,
                                                result_state,
                                                action2)

                    elif learning_policy is LearningPolicy.SARSA:
                        # Select "actual" action2
                        # (which is already None, if state is terminal)
                        q_estimator.reward(state,
                                           action,
                                           accumulated_reward,
                                           result_state,
                                           result_state_action)
                    else:
                        raise NotImplementedError('Unknown LearningPolicy')

                    action = result_state_action
                    accumulated_reward = 0

                state = result_state

                # Observe some fitness related variables
                # TODO: If the agent ever gets good enough to complete the
                # level, then world/stage will need to be added to fitness
                # too (not only x_pos)
                frames += 1
                if info['x_pos'] > max_x:
                    max_x = info['x_pos']
                    time_max_x = info['time']

            # estimator should perform batch-updates in finished() (if used)
            q_estimator.episode_finished()
            action_policy.episode_finished()
            # Record fitness variables
            # Important: stop timer *after* batch-updates for fair FPS-comparison
            time_elapsed = time.monotonic() - time_start
            # Listener
            if self.listener is not None:
                self.listener.episode_finished(
                    self.current_episode,
                    time_elapsed,
                    400 - time_max_x,
                    frames,
                    max_x)


if __name__ == '__main__':
    print('Starting MarioRLAgent *without* GUI with some default settings. ' +
          'This is probably *not* what you want unless you know what you\'re ' +
          'doing')

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
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    action_list = list(range(env.action_space.n))
    action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.05)
    learning_policy = LearningPolicy.Q
    q_estimator = TabQ.TabularQEstimator(discount=0.9, learning_rate=0.1)
    listener = PlotOnlyListener(q_estimator, action_policy, learning_policy)
    agent = MarioRLAgent(env, q_estimator, action_policy, learning_policy, listener=listener)
    agent.run()
