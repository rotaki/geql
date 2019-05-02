from enum import Enum
import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros.actions import *
import time

import TrainingStats as TS
import impl.EpsilonGreedyActionPolicy as EGAP
import impl.TabularQEstimator as TabQ

env = gym_smb.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
action_list = list(range(env.action_space.n))
action_policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.05)
q_estimator = TabQ.TabularQEstimator(discount=0.9, learning_rate=0.1)

class LearningPolicy(Enum):
    Q = 0
    SARSA = 1
    def describe(p):
        if p is LearningPolicy.Q:
            return 'Q-Learning'
        if p is LearningPolicy.SARSA:
            return 'SARSA'
        return 'Unknown'

def train(action_interval = 6, learning_policy=LearningPolicy.Q, max_episodes = 100000):
    training_stats = TS.TrainingStats(
        q_estimator,
        action_policy,
        'Learning policy: {}'.format(LearningPolicy.describe(learning_policy)))
    # Show the (empty) plot immediately
    training_stats.plot()
    for episode in range(max_episodes):
        episode_done = False
        state = env.reset()
        max_x = 0
        time_max_x = 0
        time_start = time.monotonic()
        frames = 0
        action = action_policy.get_action(state, q_estimator)
        accumulated_reward = 0
        while not episode_done:
            result_state, reward, episode_done, info = env.step(action)
            # Accumulate reward between non-action frames
            accumulated_reward += reward
            env.render()
            if frames > 0 and frames % action_interval == 0:
                # These are the only frames the RL-agent "sees"

                # Action for next state
                result_state_action = action_policy.get_action(result_state,
                                                               q_estimator)

                # Update q-estimator depends on which learning policy is used
                if learning_policy is LearningPolicy.Q:
                    # Select action2 with highest Q(s, action2)
                    action_values = q_estimator.batch_estimate(result_state,
                                                               action_list)
                    av_pair = max(action_values, key=lambda av: av[0])
                    q_estimator.reward(state, action, accumulated_reward,
                                       result_state, av_pair[0])
                elif learning_policy is LearningPolicy.SARSA:
                    # Select "actual" action2 
                    q_estimator.reward(state, action, accumulated_reward,
                                       result_state, result_state_action)
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

            # Terminate the episode on death-signal
            if reward == -15:
                episode_done = True

        # estimator should perform batch-updates in finished() (if used)
        q_estimator.episode_finished()
        action_policy.episode_finished()
        # Record fitness variables
        # Important: stop timer *after* batch-updates for fair FPS-comparison
        time_elapsed = time.monotonic() - time_start
        training_stats.add_episode_stats(time_elapsed, 400 - time_max_x, frames, max_x)
        try:
            training_stats.plot()
        except TclError as e:
            print('Warning: TclError on plot')
            

if __name__ == "__main__":
    train()
