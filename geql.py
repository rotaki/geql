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
q_estimator = TabQ.TabularQEstimator(actions=action_list,
                                discount=0.8,
                                learning_rate=0.2,
                                policy=None)
policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.05)



def train(max_stuck_time=30, max_episodes = 100000):
    training_stats = TS.TrainingStats(q_estimator,
                                      policy,
                                      None if max_stuck_time is None else
                                      'Max stuck time: {}'.format(max_stuck_time))

    # Show the (empty) plot immediately
    training_stats.plot()
    
    for episode in range(max_episodes):
        episode_done = False
        state = env.reset()
        max_x = 0
        time_max_x = 0
        time_start = time.monotonic()
        frames = 0
        while not episode_done:
            action = policy.action(state, q_estimator)
            result_state, reward, episode_done, info = env.step(action)
            # Don't know if the whole stuck-timer is actually a good idea
            if max_stuck_time is not None:
            if max_stuck_time is not None and max_stuck_time > 0:
                if info['x_pos'] < max_x and info['time'] + max_stuck_time < time_max_x:
                    #reward = -5
                    episode_done = True
                    
            q_estimator.reward(state, action, reward, result_state)
            state = result_state
            env.render()
            
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
        policy.episode_finished()
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
