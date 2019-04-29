import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros.actions import *
#from IPython.display import Image
#import RLInterfaces
#from RLInterfaces import *
import impl.EpsilonGreedyActionPolicy as EGAP
import impl.TabularQEstimator as TabQ

env = gym_smb.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
action_list = list(range(env.action_space.n))
q_estimator = TabQ.TabularQEstimator(actions=action_list,
                                discount=0.9,
                                learning_rate=0.1,
                                policy=None)
policy = EGAP.EpsilonGreedyActionPolicy(actions=action_list, epsilon=0.05)

for episode in range(1000):
    episode_done = False
    state = env.reset()
    while not episode_done:
        action = policy.action(state, q_estimator)
        result_state, reward, episode_done, info = env.step(action)
        q_estimator.reward(state, action, reward, result_state)
        env.render()
        state = result_state
